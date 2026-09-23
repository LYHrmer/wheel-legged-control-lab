// Copyright 2026. Portions follow MuJoCo 3.12.0 engine_collision_convex.c
// (Copyright 2021 DeepMind Technologies Limited, Apache License 2.0):
// witness conversion at lines 87-132, distinctness/rotation at 822-853,
// and perturbation order at 881-961. Official source bytes are frozen beside
// this diagnostic. This is a copied wrapper around exported native CCD APIs,
// not mjc_Convex, a model, or a replay of coupled dynamics.
#define _GNU_SOURCE
#include <mujoco/mujoco.h>
#include "engine/engine_collision_gjk.h"
#include "d1_contact_native_ccd_input.h"

#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum { MAX_PAIR_CONTACTS=5, MAX_MATH_HELPERS=512, MAX_CCD_ATTEMPTS=16 };
static const char* slot_names[5]={"primary","axis0_negative","axis0_positive",
                                  "axis1_negative","axis1_positive"};
typedef struct {
  FILE* log;
  void* raw_workspace;
  void* aligned_workspace;
  size_t workspace_bytes;
  int init_calls, size_calls, malloc_calls, free_calls;
  int version_calls, version_string_calls, ccd_attempts, ccd_returns;
  int math_total, math_normalize, math_dot, math_cross, math_dist;
  int math_axisangle, math_quatmat, math_matvec;
  int local_frame_calls, local_rotation_calls, local_matmul_calls;
  int error;
} Run;
typedef struct {
  mjPreContact contact;
  mjtNum raw_dist;
  int slot, accepted, distinct, support_valid, reference_equal, reference_corresponds;
  mjtNum frame[9], residual, ref_delta_pos[3], ref_delta_dist, ref_normal_dot;
} Candidate;
typedef struct {
  int count, all_finite, all_support_valid, all_reference_correspond;
  int all_reference_equal, primary_valid, invalid_count, bad_index5_matched;
  Candidate candidates[MAX_PAIR_CONTACTS];
} CaseResult;

static volatile int warning_seen=0;
static void capture_warning(const char* message) {
  warning_seen=1;
  fprintf(stderr,"MuJoCo warning: %s\n",message?message:"(null)");
  fflush(stderr);
}
static int persist(Run* r) {
  return fputc('\n',r->log)!=EOF && fflush(r->log)==0 && fsync(fileno(r->log))==0;
}
static void json_num(FILE* f, mjtNum x) {
  if (isfinite((double)x)) fprintf(f,"%.17g",(double)x);
  else fputs("null",f);
}
static void vec(FILE* f, const mjtNum* x, int n) {
  fputc('[',f);
  for (int i=0;i<n;i++) {
    if (i) fputc(',',f);
    json_num(f,x[i]);
  }
  fputc(']',f);
}
static void hexvec(FILE* f, const mjtNum* x, int n) {
  fputc('[',f);
  for (int i=0;i<n;i++) fprintf(f,"%s\"%a\"",i?",":"",(double)x[i]);
  fputc(']',f);
}
static int finite_vec(const mjtNum* x, int n) {
  for (int i=0;i<n;i++) if (!isfinite((double)x[i])) return 0;
  return 1;
}
static int equal_bits(const mjtNum* x, const double* y, int n) {
  return sizeof(mjtNum)==sizeof(double) && memcmp(x,y,(size_t)n*sizeof(double))==0;
}
static int math_charge(Run* r, int* counter) {
  ++r->math_total; ++*counter;
  if (r->math_total>MAX_MATH_HELPERS) { r->error=1; return 0; }
  return 1;
}
static mjtNum counted_normalize(Run* r, mjtNum x[3]) {
  if (!math_charge(r,&r->math_normalize)) return 0;
  return mju_normalize3(x);
}
static mjtNum counted_dot(Run* r, const mjtNum a[3], const mjtNum b[3]) {
  if (!math_charge(r,&r->math_dot)) return 0;
  return mju_dot3(a,b);
}
static mjtNum counted_dist(Run* r, const mjtNum a[3], const mjtNum b[3]) {
  if (!math_charge(r,&r->math_dist)) return 0;
  return mju_dist3(a,b);
}
static void counted_cross(Run* r, mjtNum out[3], const mjtNum a[3], const mjtNum b[3]) {
  if (math_charge(r,&r->math_cross)) mju_cross(out,a,b);
}
static void counted_axisangle(Run* r, mjtNum out[4], const mjtNum axis[3], mjtNum angle) {
  if (math_charge(r,&r->math_axisangle)) mju_axisAngle2Quat(out,axis,angle);
}
static void counted_quatmat(Run* r, mjtNum out[9], const mjtNum quat[4]) {
  if (math_charge(r,&r->math_quatmat)) mju_quat2Mat(out,quat);
}
static void counted_matvec(Run* r, mjtNum out[3], const mjtNum mat[9], const mjtNum x[3]) {
  if (math_charge(r,&r->math_matvec)) mju_mulMatVec3(out,mat,x);
}

// Source-equivalent arithmetic for mju_makeFrame (not public in mujoco.h).
static int comparison_frame(Run* r, const mjtNum normal[3], mjtNum frame[9]) {
  ++r->local_frame_calls;
  memcpy(frame,normal,3*sizeof(mjtNum)); memset(frame+3,0,6*sizeof(mjtNum));
  if (counted_normalize(r,frame)<0.5) return 0;
  if (counted_dot(r,frame+3,frame+3)<0.25) {
    if (frame[1]<0.5 && frame[1]>-0.5) frame[4]=1;
    else frame[5]=1;
  }
  mjtNum dot=counted_dot(r,frame,frame+3);
  for (int i=0;i<3;i++) frame[3+i]-=frame[i]*dot;
  if (counted_normalize(r,frame+3)<=0) return 0;
  counted_cross(r,frame+6,frame,frame+3);
  return !r->error && finite_vec(frame,9);
}
static void local_matmul(Run* r, mjtNum out[9], const mjtNum a[9], const mjtNum b[9]) {
  ++r->local_matmul_calls;
  for (int i=0;i<3;i++) for (int j=0;j<3;j++) {
    out[3*i+j]=a[3*i]*b[j]+a[3*i+1]*b[3+j]+a[3*i+2]*b[6+j];
  }
}
// engine_collision_convex.c:834-851; local matrix product, public matvec.
static void local_rotate(Run* r, const mjtNum origin[3], const mjtNum rot[9],
                         mjtNum mat[9], mjtNum pos[3]) {
  ++r->local_rotation_calls;
  mjtNum product[9], rel[3], displacement[3];
  local_matmul(r,product,rot,mat); memcpy(mat,product,sizeof(product));
  for (int i=0;i<3;i++) rel[i]=origin[i]-pos[i];
  counted_matvec(r,displacement,rot,rel);
  for (int i=0;i<3;i++) {
    displacement[i]-=rel[i]; pos[i]-=displacement[i];
  }
}
static int support_valid(const KernelPose* p, const mjPreContact* c,
                         const mjtNum frame[9], mjtNum* residual) {
  mjtNum tol=fabs(c->dist)+fmax(kPairMargin,fmax(kWheelMargin,kBoxMargin))+1e-7;
  mjtNum squared=0; int found=0,inside=1;
  for (int i=0;i<3;i++) {
    mjtNum n=-frame[i],center=p->box_pos[i],half=kBoxSize[i];
    int face=fabs(n)>1e-6 && fabs(c->pos[i]-(center+copysign(half,n)))<=tol;
    if (face) found=1; else squared+=n*n;
    if (fabs(c->pos[i]-center)>half+tol) inside=0;
  }
  *residual=sqrt(squared);
  return found && inside && *residual<=0.0021;
}
static int reference_match(const KernelReference* ref, Candidate* c) {
  mjtNum squared=0;
  c->ref_normal_dot=0;
  for (int i=0;i<3;i++) {
    c->ref_delta_pos[i]=c->contact.pos[i]-ref->pos[i];
    squared+=c->ref_delta_pos[i]*c->ref_delta_pos[i];
    c->ref_normal_dot+=c->frame[i]*ref->frame[i];
  }
  c->ref_delta_dist=c->contact.dist-ref->dist;
  c->reference_equal=equal_bits(c->contact.pos,ref->pos,3)
    && memcmp(&c->contact.dist,&ref->dist,sizeof(double))==0
    && equal_bits(c->frame,ref->frame,9);
  mjtNum limit=fabs(ref->dist)+kPairMargin+1e-7;
  c->reference_corresponds=sqrt(squared)<=limit
    && fabs(c->ref_delta_dist)<=limit && c->ref_normal_dot>0.99;
  return c->reference_corresponds;
}

static void log_descriptor(FILE* f, const mjCCDObj* obj) {
  fprintf(f,"{\"geom\":%d,\"geom_type\":%d,\"margin\":",
          obj->geom,obj->geom_type);
  json_num(f,obj->margin);
  fputs(",\"size\":",f);
  vec(f,obj->size,3); fprintf(f,",\"pos\":"); vec(f,obj->pos,3);
  fprintf(f,",\"pos_hex\":"); hexvec(f,obj->pos,3);
  fprintf(f,",\"mat\":"); vec(f,obj->mat,9);
  fprintf(f,",\"mat_hex\":"); hexvec(f,obj->mat,9);
  fprintf(f,",\"vertindex\":%d,\"meshindex\":%d}",obj->vertindex,obj->meshindex);
}
static int input_unchanged(const mjCCDObj* a, const mjCCDObj* b,
                           const KernelPose* p) {
  return equal_bits(a->pos,p->wheel_pos,3) && equal_bits(a->mat,p->wheel_mat,9)
    && equal_bits(b->pos,p->box_pos,3) && equal_bits(b->mat,p->box_mat,9)
    && equal_bits(a->size,kWheelSize,3) && equal_bits(b->size,kBoxSize,3)
    && a->margin==kPairMargin && b->margin==kPairMargin;
}

static int log_status(Run* r, const mjCCDStatus* s, mjtNum ret,
                      const mjCCDObj* a, const mjCCDObj* b,
                      int attempt_seq, int case_no, int native_index, int slot) {
  int valid_bounds=s->nx>=0 && s->nx<=1 && s->nsimplex>=0 && s->nsimplex<=4;
  int valid_witness=valid_bounds && ret<0 && s->nx==1;
  int status_finite=valid_bounds && isfinite((double)ret) && isfinite((double)s->tolerance)
    && isfinite((double)s->dist_cutoff)
    && s->max_iterations==35 && s->max_contacts==1 && s->separated>=0 && s->separated<=1
    && s->gjk_iterations>=0 && s->gjk_iterations<=35
    && s->epa_iterations>=0 && s->epa_iterations<=35;
  for (int i=0;valid_bounds && i<s->nsimplex;i++) {
    status_finite &= finite_vec(s->simplex[i].vert,3)
      && finite_vec(s->simplex[i].vert1,3) && finite_vec(s->simplex[i].vert2,3);
  }
  if (valid_witness) status_finite &= isfinite((double)s->dist[0])
    && finite_vec(s->x1,3) && finite_vec(s->x2,3);
  status_finite &= finite_vec(a->pos,3) && finite_vec(a->mat,9)
    && finite_vec(b->pos,3) && finite_vec(b->mat,9)
    && finite_vec(a->size,3) && finite_vec(b->size,3)
    && isfinite((double)a->margin) && isfinite((double)b->margin);
  fprintf(r->log,"{\"event\":\"ccd_return\",\"attempt_seq\":%d,"
          "\"case_no\":%d,\"native_index\":%d,\"slot\":%d,\"slot_name\":\"%s\","
          "\"return_distance\":",attempt_seq,case_no,native_index,slot,slot_names[slot]);
  if (isfinite((double)ret)) fprintf(r->log,"%.17g",(double)ret); else fputs("null",r->log);
  fprintf(r->log,",\"return_distance_hex\":\"%a\",\"bounds_valid\":%s,"
          "\"witness_valid\":%s,\"status_finite\":%s,\"warning_seen\":%s,\"status\":{"
          "\"nx\":%d,\"nsimplex\":%d,\"separated\":%d,"
          "\"max_iterations\":%d,\"tolerance\":",
          (double)ret,valid_bounds?"true":"false",valid_witness?"true":"false",
          status_finite?"true":"false",warning_seen?"true":"false",
          s->nx,s->nsimplex,s->separated,s->max_iterations);
  json_num(r->log,s->tolerance);
  fprintf(r->log,",\"tolerance_hex\":\"%a\",\"max_contacts\":%d,"
          "\"dist_cutoff\":",(double)s->tolerance,s->max_contacts);
  json_num(r->log,s->dist_cutoff);
  fprintf(r->log,",\"dist_cutoff_hex\":\"%a\",\"raw_status_dist0\":",
          (double)s->dist_cutoff);
  json_num(r->log,s->dist[0]);
  fprintf(r->log,",\"raw_status_dist0_hex\":\"%a\","
          "\"raw_status_x1\":",(double)s->dist[0]);
  vec(r->log,s->x1,3); fprintf(r->log,",\"raw_status_x1_hex\":");
  hexvec(r->log,s->x1,3);
  fprintf(r->log,",\"raw_status_x2\":"); vec(r->log,s->x2,3);
  fprintf(r->log,",\"raw_status_x2_hex\":"); hexvec(r->log,s->x2,3);
  fprintf(r->log,",\"gjk_iterations\":%d,\"epa_iterations\":%d,"
          "\"epa_status\":%d,\"simplex\":[",
          s->gjk_iterations,s->epa_iterations,(int)s->epa_status);
  for (int i=0;i<s->nsimplex && i<4;i++) {
    const Vertex* v=&s->simplex[i];
    fprintf(r->log,"%s{\"vert\":",i?",":""); vec(r->log,v->vert,3);
    fprintf(r->log,",\"vert_hex\":"); hexvec(r->log,v->vert,3);
    fprintf(r->log,",\"vert1\":"); vec(r->log,v->vert1,3);
    fprintf(r->log,",\"vert1_hex\":"); hexvec(r->log,v->vert1,3);
    fprintf(r->log,",\"vert2\":"); vec(r->log,v->vert2,3);
    fprintf(r->log,",\"vert2_hex\":"); hexvec(r->log,v->vert2,3);
    fprintf(r->log,",\"index1\":%d,\"index2\":%d}",v->index1,v->index2);
  }
  fprintf(r->log,"],\"witnesses\":[");
  if (valid_witness) {
    fputs("{\"dist\":",r->log); json_num(r->log,s->dist[0]);
    fprintf(r->log,",\"dist_hex\":\"%a\",\"x1\":",(double)s->dist[0]);
    vec(r->log,s->x1,3); fprintf(r->log,",\"x2\":"); vec(r->log,s->x2,3);
    fprintf(r->log,",\"x1_hex\":"); hexvec(r->log,s->x1,3);
    fprintf(r->log,",\"x2_hex\":"); hexvec(r->log,s->x2,3);
    fputc('}',r->log);
  }
  fprintf(r->log,"]},\"descriptor_after\":[");
  log_descriptor(r->log,a); fputc(',',r->log); log_descriptor(r->log,b);
  fputs("]}",r->log);
  return persist(r) && valid_bounds && status_finite && !warning_seen;
}

static int query(Run* r, mjCCDConfig* config, mjCCDObj* a, mjCCDObj* b,
                 const KernelPose* pose, int slot, int case_no,
                 mjPreContact* out, mjtNum* raw_dist) {
  if (r->ccd_attempts>=MAX_CCD_ATTEMPTS) return -1;
  int seq=++r->ccd_attempts;
  fprintf(r->log,"{\"event\":\"ccd_attempt\",\"attempt_seq\":%d,"
          "\"case_no\":%d,\"native_index\":%d,\"slot\":%d,\"slot_name\":\"%s\","
          "\"config\":{\"max_iterations\":%d,\"tolerance\":%.17g,"
          "\"max_contacts\":%d,\"dist_cutoff\":%.17g,\"npolygonmax\":%d,"
          "\"nmeshdegmax\":%d},\"descriptor_before\":[",
          seq,case_no,pose->native_index,slot,slot_names[slot],config->max_iterations,
          (double)config->tolerance,config->max_contacts,(double)config->dist_cutoff,
          config->npolygonmax,config->nmeshdegmax);
  log_descriptor(r->log,a); fputc(',',r->log); log_descriptor(r->log,b);
  fputs("]}",r->log);
  if (!persist(r)) return -1;  // No call without a durable attempted record.

  mjCCDStatus status={0};
  warning_seen=0;
  mjtNum ret=mjc_ccd(config,&status,a,b);
  ++r->ccd_returns;
  if (!log_status(r,&status,ret,a,b,seq,case_no,pose->native_index,slot)) return -1;
  if (status.gjk_iterations>35 || status.epa_iterations>35
      || !finite_vec(status.x1,3*(status.nx>0?status.nx:0))
      || !finite_vec(status.x2,3*(status.nx>0?status.nx:0))) return -1;
  if (ret>=0 || status.nx==0) return 0;
  if (status.nx!=1 || !isfinite((double)status.dist[0])) return -1;
  memset(out,0,sizeof(*out));
  *raw_dist=kPairMargin+status.dist[0];
  out->dist=*raw_dist;
  for (int i=0;i<3;i++) {
    out->pos[i]=(status.x1[i]+status.x2[i])*0.5;
    out->normal[i]=status.x1[i]-status.x2[i];
  }
  if (counted_normalize(r,out->normal)<=0 || r->error
      || !finite_vec(out->normal,3) || !finite_vec(out->pos,3)) return -1;
  return 1;
}

static void log_candidate(Run* r, const KernelPose* p, int case_no,
                          int ordinal, const Candidate* c) {
  fprintf(r->log,"{\"event\":\"candidate_decision\",\"case_no\":%d,"
          "\"native_index\":%d,\"slot\":%d,\"candidate_ordinal\":%d,"
          "\"distinct\":%s,\"accepted\":%s,\"raw_distance_m\":%.17g,"
          "\"final_distance_m\":%.17g,\"position\":",
          case_no,p->native_index,c->slot,ordinal,c->distinct?"true":"false",
          c->accepted?"true":"false",(double)c->raw_dist,(double)c->contact.dist);
  vec(r->log,c->contact.pos,3);
  fprintf(r->log,",\"raw_kernel_normal\":"); vec(r->log,c->contact.normal,3);
  fprintf(r->log,",\"raw_kernel_normal_hex\":"); hexvec(r->log,c->contact.normal,3);
  fprintf(r->log,",\"raw_kernel_tangent\":"); vec(r->log,c->contact.tangent,3);
  fprintf(r->log,",\"comparison_frame\":"); vec(r->log,c->frame,9);
  fprintf(r->log,",\"box_support_valid\":%s,\"cone_residual\":%.17g,"
          "\"reference_bitwise_equal\":%s,\"reference_corresponds\":%s,"
          "\"reference_pos_delta\":",c->support_valid?"true":"false",
          (double)c->residual,c->reference_equal?"true":"false",
          c->reference_corresponds?"true":"false");
  vec(r->log,c->ref_delta_pos,3);
  fprintf(r->log,",\"reference_dist_delta\":%.17g,\"reference_normal_dot\":%.17g}",
          (double)c->ref_delta_dist,(double)c->ref_normal_dot);
}

static int case_run(Run* r, const KernelPose* pose, int case_no,
                    int primary_only, CaseResult* result) {
  memset(result,0,sizeof(*result));
  int geom_type[2]={mjGEOM_CYLINDER,mjGEOM_BOX};
  mjtNum geom_size[6],geom_pos[6],geom_mat[18];
  memcpy(geom_size,kWheelSize,3*sizeof(double));
  memcpy(geom_size+3,kBoxSize,3*sizeof(double));
  memcpy(geom_pos,pose->wheel_pos,3*sizeof(double));
  memcpy(geom_pos+3,pose->box_pos,3*sizeof(double));
  memcpy(geom_mat,pose->wheel_mat,9*sizeof(double));
  memcpy(geom_mat+9,pose->box_mat,9*sizeof(double));
  mjModel carrier_model={0}; mjData carrier_data={0};
  carrier_model.geom_type=geom_type;
  carrier_model.geom_size=geom_size;
  carrier_data.geom_xpos=geom_pos;
  carrier_data.geom_xmat=geom_mat;
  mjCCDObj wheel={0},box={0};
  if (r->init_calls+2>8) return 0;
  ++r->init_calls; mjc_initCCDObj(&wheel,&carrier_model,&carrier_data,0,kPairMargin);
  ++r->init_calls; mjc_initCCDObj(&box,&carrier_model,&carrier_data,1,kPairMargin);
  if (warning_seen || wheel.geom_type!=mjGEOM_CYLINDER || box.geom_type!=mjGEOM_BOX
      || !wheel.support || !box.support || !wheel.center || !box.center
      || !input_unchanged(&wheel,&box,pose)) return 0;
  fprintf(r->log,"{\"event\":\"descriptors_initialized\",\"case_no\":%d,"
          "\"native_index\":%d,\"init_calls_total\":%d,\"primary_only\":%s,"
          "\"rbound\":[%.17g,%.17g],\"descriptors\":[",case_no,pose->native_index,
          r->init_calls,primary_only?"true":"false",kWheelRbound,kBoxRbound);
  log_descriptor(r->log,&wheel); fputc(',',r->log); log_descriptor(r->log,&box);
  fputs("]}",r->log);
  if (!persist(r)) return 0;

  mjCCDConfig config={0};
  config.max_iterations=35; config.tolerance=1e-6; config.max_contacts=1;
  config.dist_cutoff=0; config.npolygonmax=0; config.nmeshdegmax=0;
  config.buffer=r->aligned_workspace;
  mjPreContact first={0}; mjtNum raw_dist=0;
  int got=query(r,&config,&wheel,&box,pose,0,case_no,&first,&raw_dist);
  if (got<0) return 0;
  if (got==1) {
    Candidate* c=&result->candidates[0];
    c->contact=first; c->raw_dist=raw_dist; c->slot=0;
    c->accepted=1; c->distinct=1;
    if (!comparison_frame(r,c->contact.normal,c->frame)) return 0;
    c->support_valid=support_valid(pose,&c->contact,c->frame,&c->residual);
    if (!primary_only && pose->nref>0) reference_match(&pose->refs[0],c);
    result->count=1;
    log_candidate(r,pose,case_no,0,c); if (!persist(r)) return 0;
  }
  if (got==1 && !primary_only) {
    mjtNum axes[2][3];
    memcpy(axes[0],result->candidates[0].frame+3,3*sizeof(mjtNum));
    memcpy(axes[1],result->candidates[0].frame+6,3*sizeof(mjtNum));
    const mjtNum angles[2]={-0.001,0.001};
    for (int axis=0;axis<2;axis++) for (int sign=0;sign<2;sign++) {
      int slot=1+2*axis+sign;
      mjtNum quat[4],rot[9],inverse[9];
      counted_axisangle(r,quat,axes[axis],angles[sign]);
      counted_quatmat(r,rot,quat);
      if (r->error || !finite_vec(rot,9)) return 0;
      for (int i=0;i<3;i++) for (int j=0;j<3;j++) inverse[3*i+j]=rot[3*j+i];
      local_rotate(r,first.pos,rot,wheel.mat,wheel.pos);
      local_rotate(r,first.pos,inverse,box.mat,box.pos);
      if (r->error || !finite_vec(wheel.pos,3) || !finite_vec(wheel.mat,9)
          || !finite_vec(box.pos,3) || !finite_vec(box.mat,9)) return 0;
      mjPreContact extra={0}; mjtNum extra_raw_dist=0;
      int found=query(r,&config,&wheel,&box,pose,slot,case_no,&extra,&extra_raw_dist);
      // Restore only pos/mat, preserving the support-cache vertindex state.
      memcpy(wheel.pos,geom_pos,3*sizeof(mjtNum));
      memcpy(wheel.mat,geom_mat,9*sizeof(mjtNum));
      memcpy(box.pos,geom_pos+3,3*sizeof(mjtNum));
      memcpy(box.mat,geom_mat+9,9*sizeof(mjtNum));
      if (found<0) return 0;
      if (found==1) {
        mjtNum tolerance=1e-3*fmin(kWheelRbound,kBoxRbound);
        int distinct=1;
        for (int i=0;i<result->count;i++) {
          if (counted_dist(r,result->candidates[i].contact.pos,extra.pos)<=tolerance) distinct=0;
        }
        Candidate candidate={0};
        candidate.contact=extra; candidate.raw_dist=extra_raw_dist;
        candidate.slot=slot; candidate.distinct=distinct; candidate.accepted=distinct;
        if (distinct && result->count<MAX_PAIR_CONTACTS) {
          candidate.contact.dist=first.dist;
        } else candidate.accepted=0;
        if (!comparison_frame(r,candidate.contact.normal,candidate.frame)) return 0;
        candidate.support_valid=support_valid(pose,&candidate.contact,candidate.frame,
                                             &candidate.residual);
        if (candidate.accepted && !primary_only && result->count<pose->nref)
          reference_match(&pose->refs[result->count],&candidate);
        if (candidate.accepted) {
          result->candidates[result->count]=candidate;
          ++result->count;
        }
        log_candidate(r,pose,case_no,result->count-(candidate.accepted?1:0),&candidate);
        if (!persist(r) || r->error) return 0;
      } else {
        fprintf(r->log,"{\"event\":\"candidate_decision\",\"case_no\":%d,"
                "\"native_index\":%d,\"slot\":%d,\"found\":false,\"accepted\":false}",
                case_no,pose->native_index,slot);
        if (!persist(r)) return 0;
      }
    }
  }
  if (!input_unchanged(&wheel,&box,pose)
      || !equal_bits(geom_pos,pose->wheel_pos,3)
      || !equal_bits(geom_pos+3,pose->box_pos,3)
      || !equal_bits(geom_mat,pose->wheel_mat,9)
      || !equal_bits(geom_mat+9,pose->box_mat,9)) return 0;
  result->all_finite=1;
  result->all_support_valid=1;
  result->all_reference_correspond=!primary_only && result->count==pose->nref;
  result->all_reference_equal=!primary_only && result->count==pose->nref;
  for (int i=0;i<result->count;i++) {
    Candidate* c=&result->candidates[i];
    if (!finite_vec(c->contact.pos,3) || !finite_vec(c->contact.normal,3)
        || !isfinite((double)c->contact.dist)) result->all_finite=0;
    result->all_support_valid &= c->support_valid;
    result->invalid_count += !c->support_valid;
    if (i==0) result->primary_valid=c->support_valid;
    if (!primary_only && i<pose->nref) {
      reference_match(&pose->refs[i],c);
      result->all_reference_correspond &= c->reference_corresponds;
      result->all_reference_equal &= c->reference_equal;
      if (pose->native_index==3486 && pose->refs[i].original_index==5
          && !c->support_valid && -c->frame[2]<-0.9 && c->reference_corresponds)
        result->bad_index5_matched=1;
    }
  }
  fprintf(r->log,"{\"event\":\"case_complete\",\"case_no\":%d,"
          "\"native_index\":%d,\"primary_only\":%s,\"count\":%d,"
          "\"all_finite\":%s,\"all_support_valid\":%s,\"primary_valid\":%s,"
          "\"invalid_count\":%d,\"bad_index5_matched\":%s,"
          "\"all_reference_correspond\":%s,\"all_reference_equal\":%s,"
          "\"input_unchanged\":true,\"final_candidates\":[",
          case_no,pose->native_index,primary_only?"true":"false",result->count,
          result->all_finite?"true":"false",result->all_support_valid?"true":"false",
          result->primary_valid?"true":"false",result->invalid_count,
          result->bad_index5_matched?"true":"false",
          result->all_reference_correspond?"true":"false",
          result->all_reference_equal?"true":"false");
  for (int i=0;i<result->count;i++) {
    Candidate* c=&result->candidates[i];
    fprintf(r->log,"%s{\"index\":%d,\"old_contact_index\":%d,"
            "\"slot\":%d,\"dist\":%.17g,\"raw_dist\":%.17g,\"pos\":",
            i?",":"",i,i<pose->nref?pose->refs[i].original_index:-1,c->slot,
            (double)c->contact.dist,(double)c->raw_dist);
    vec(r->log,c->contact.pos,3); fprintf(r->log,",\"normal\":");
    vec(r->log,c->contact.normal,3); fprintf(r->log,",\"tangent\":");
    vec(r->log,c->contact.tangent,3); fprintf(r->log,",\"frame\":");
    vec(r->log,c->frame,9);
    fprintf(r->log,",\"support_valid\":%s,\"reference_equal\":%s,"
            "\"reference_corresponds\":%s,\"reference_pos_delta\":",
            c->support_valid?"true":"false",c->reference_equal?"true":"false",
            c->reference_corresponds?"true":"false");
    vec(r->log,c->ref_delta_pos,3);
    fprintf(r->log,",\"reference_dist_delta\":%.17g,\"reference_normal_dot\":%.17g}",
            (double)c->ref_delta_dist,(double)c->ref_normal_dot);
  }
  fputs("]}",r->log);
  return persist(r) && result->all_finite && !r->error && !warning_seen;
}

static int log_summary(Run* r, const char* result, int gate_fourth) {
  fprintf(r->log,"{\"event\":\"summary\",\"result\":\"%s\","
          "\"fourth_allowed\":%s,\"init_calls\":%d,\"ccd_size_calls\":%d,"
          "\"workspace_malloc_calls\":%d,\"workspace_free_calls\":%d,"
          "\"ccd_attempts\":%d,\"ccd_returns\":%d,\"version_calls\":%d,"
          "\"version_string_calls\":%d,\"math_total\":%d,\"math_by_name\":{"
          "\"mju_normalize3\":%d,\"mju_dot3\":%d,\"mju_cross\":%d,"
          "\"mju_dist3\":%d,\"mju_axisAngle2Quat\":%d,\"mju_quat2Mat\":%d,"
          "\"mju_mulMatVec3\":%d},\"local_math\":{\"frame\":%d,"
          "\"rotate\":%d,\"matmul\":%d},\"new_control\":0,\"new_native\":0,"
          "\"model_compiles\":0,\"model_allocations\":0,\"data_allocations\":0,"
          "\"qualification_granted\":false,\"original_score_overridden\":false,"
          "\"rl_gate_open\":false}",result,gate_fourth?"true":"false",
          r->init_calls,r->size_calls,r->malloc_calls,r->free_calls,r->ccd_attempts,
          r->ccd_returns,r->version_calls,r->version_string_calls,r->math_total,
          r->math_normalize,r->math_dot,r->math_cross,r->math_dist,r->math_axisangle,
          r->math_quatmat,r->math_matvec,r->local_frame_calls,r->local_rotation_calls,
          r->local_matmul_calls);
  return persist(r);
}

int main(int argc, char** argv) {
  if (argc!=2) {
    fprintf(stderr,"usage: %s EXISTING_EMPTY_OUTPUT_DIR\n",argv[0]); return 2;
  }
  char path[4096];
  if (snprintf(path,sizeof(path),"%s/ccd_events.ndjson",argv[1])>=(int)sizeof(path)) return 2;
  int fd=open(path,O_CREAT|O_EXCL|O_WRONLY,0644);
  if (fd<0) { perror("exclusive CCD event log"); return 2; }
  Run r={0}; r.log=fdopen(fd,"w");
  if (!r.log) { close(fd); return 2; }
  int exitcode=2, fourth_allowed=0;
  const char* classification="preflight_error";
  void (*prior_warning)(const char*)=mju_user_warning;
  mju_user_warning=capture_warning;
  Dl_info library={0},initializer={0},sizesymbol={0},ccdsymbol={0};
  char actual_path[4096];
  if (sizeof(mjtNum)!=8 || !dladdr((void*)(uintptr_t)&mjc_ccd,&library)
      || !dladdr((void*)(uintptr_t)&mjc_initCCDObj,&initializer)
      || !dladdr((void*)(uintptr_t)&mjc_ccdSize,&sizesymbol)
      || !dladdr((void*)(uintptr_t)&mjc_ccd,&ccdsymbol)
      || !library.dli_fname || !realpath(library.dli_fname,actual_path)
      || strcmp(actual_path,KERNEL_EXPECTED_LIB_PATH)!=0
      || strcmp(library.dli_fname,initializer.dli_fname)!=0
      || strcmp(library.dli_fname,sizesymbol.dli_fname)!=0
      || strcmp(library.dli_fname,ccdsymbol.dli_fname)!=0) goto finish;
  ++r.version_calls; int version=mj_version();
  ++r.version_string_calls; const char* version_string=mj_versionString();
  if (version!=mjVERSION_HEADER || version!=3012000
      || !version_string || strcmp(version_string,"3.12.0")!=0) goto finish;
  fprintf(r.log,"{\"event\":\"preflight\",\"loaded_library_path\":\"%s\","
          "\"expected_library_sha256\":\"%s\",\"version\":%d,"
          "\"version_string\":\"%s\",\"fixture_sha256\":\"%s\","
          "\"contract_sha256\":\"%s\",\"mjtNum_bytes\":%zu,"
          "\"symbols_from_fixed_library\":true,\"model_or_data_allocated\":false}",
          actual_path,KERNEL_EXPECTED_LIB_SHA256,version,version_string,
          KERNEL_FIXTURE_SHA256,KERNEL_CONTRACT_02_SHA256,sizeof(mjtNum));
  if (!persist(&r)) goto finish;
  ++r.size_calls;
  r.workspace_bytes=mjc_ccdSize(0,0,35);
  if (!r.workspace_bytes || r.workspace_bytes>100000000) goto finish;
  ++r.malloc_calls;
  r.raw_workspace=malloc(r.workspace_bytes+64);
  if (!r.raw_workspace) goto finish;
  r.aligned_workspace=(void*)(((uintptr_t)r.raw_workspace+63u)&~(uintptr_t)63u);
  fprintf(r.log,"{\"event\":\"workspace\",\"ccd_size_calls\":1,"
          "\"bytes\":%zu,\"aligned_to_bytes\":64,\"malloc_calls\":1}",r.workspace_bytes);
  if (!persist(&r)) goto finish;

  CaseResult cases[4];
  for (int i=0;i<3;i++) {
    if (!case_run(&r,&kPoses[i],i+1,0,&cases[i])) {
      classification="case_error"; exitcode=3; goto finish;
    }
  }
  fourth_allowed=cases[0].count==kPoses[0].nref && cases[2].count==kPoses[2].nref
    && cases[0].all_support_valid && cases[2].all_support_valid
    && cases[0].all_reference_correspond && cases[2].all_reference_correspond
    && cases[1].count==kPoses[1].nref && cases[1].primary_valid
    && cases[1].invalid_count==1 && cases[1].bad_index5_matched
    && cases[1].all_reference_correspond && cases[0].all_finite
    && cases[1].all_finite && cases[2].all_finite && !warning_seen;
  fprintf(r.log,"{\"event\":\"fourth_gate\",\"allowed\":%s,"
          "\"neighbor_counts_and_geometry\":%s,\"neighbor_correspondence\":%s,"
          "\"target_count_primary_bad5\":%s,\"target_all_correspond\":%s}",
          fourth_allowed?"true":"false",
          (cases[0].count==3 && cases[2].count==3 && cases[0].all_support_valid
           && cases[2].all_support_valid)?"true":"false",
          (cases[0].all_reference_correspond && cases[2].all_reference_correspond)?"true":"false",
          (cases[1].count==3 && cases[1].primary_valid && cases[1].invalid_count==1
           && cases[1].bad_index5_matched)?"true":"false",
          cases[1].all_reference_correspond?"true":"false");
  if (!persist(&r)) { classification="io_error"; exitcode=3; goto finish; }
  if (fourth_allowed) {
    if (!case_run(&r,&kPoses[1],4,1,&cases[3])) {
      classification="fourth_case_error"; exitcode=3; goto finish;
    }
  }
  classification=fourth_allowed?
    (cases[0].all_reference_equal && cases[1].all_reference_equal
     && cases[2].all_reference_equal?"bitwise_reproduction":"same_geometry_anomaly"):
    "reproduction_gap";
  exitcode=0;

finish:
  if (r.raw_workspace) { free(r.raw_workspace); ++r.free_calls; }
  mju_user_warning=prior_warning;
  if (!log_summary(&r,classification,fourth_allowed)) exitcode=3;
  fclose(r.log);
  return exitcode;
}
