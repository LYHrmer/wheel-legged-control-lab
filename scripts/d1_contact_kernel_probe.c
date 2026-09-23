/* Bounded two-primitive MuJoCo 3.12.0 narrowphase probe.
 * Compile with the generated input header and the pinned installed library.
 * This program never steps, forwards, resets, or evaluates dynamics.
 */
#define _GNU_SOURCE
#include <mujoco/mujoco.h>

#include <dlfcn.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "d1_contact_kernel_input.h"

#define MAX_PAIR_CONTACTS 5

/* Local SHA-256 avoids a second library dependency and checks the loaded file. */
static const uint32_t sha_k[64] = {
  0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
  0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
  0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
  0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
  0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
  0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
  0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
  0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};
typedef struct { uint32_t h[8]; uint64_t nbytes; unsigned char block[64]; size_t used; } Sha256;
static uint32_t rotr(uint32_t x, unsigned n) { return (x >> n) | (x << (32 - n)); }
static void sha_init(Sha256* s) {
  const uint32_t h[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                         0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
  memcpy(s->h, h, sizeof(h)); s->nbytes = 0; s->used = 0;
}
static void sha_block(Sha256* s) {
  uint32_t w[64];
  for (int i=0; i<16; ++i) {
    w[i] = ((uint32_t)s->block[4*i]<<24) | ((uint32_t)s->block[4*i+1]<<16)
         | ((uint32_t)s->block[4*i+2]<<8) | s->block[4*i+3];
  }
  for (int i=16; i<64; ++i) {
    uint32_t a=w[i-15], b=w[i-2];
    w[i] = w[i-16] + (rotr(a,7)^rotr(a,18)^(a>>3)) + w[i-7]
         + (rotr(b,17)^rotr(b,19)^(b>>10));
  }
  uint32_t a=s->h[0],b=s->h[1],c=s->h[2],d=s->h[3];
  uint32_t e=s->h[4],f=s->h[5],g=s->h[6],h=s->h[7];
  for (int i=0; i<64; ++i) {
    uint32_t t1=h+(rotr(e,6)^rotr(e,11)^rotr(e,25))+((e&f)^(~e&g))+sha_k[i]+w[i];
    uint32_t t2=(rotr(a,2)^rotr(a,13)^rotr(a,22))+((a&b)^(a&c)^(b&c));
    h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
  }
  s->h[0]+=a; s->h[1]+=b; s->h[2]+=c; s->h[3]+=d;
  s->h[4]+=e; s->h[5]+=f; s->h[6]+=g; s->h[7]+=h;
}
static void sha_update(Sha256* s, const unsigned char* data, size_t len) {
  s->nbytes += len;
  for (size_t i=0; i<len; ++i) {
    s->block[s->used++] = data[i];
    if (s->used == 64) { sha_block(s); s->used = 0; }
  }
}
static void sha_final(Sha256* s, char out[65]) {
  uint64_t bits = s->nbytes * 8;
  unsigned char one=0x80, zero=0;
  sha_update(s, &one, 1);
  while (s->used != 56) sha_update(s, &zero, 1);
  unsigned char length[8];
  for (int i=0; i<8; ++i) length[7-i]=(unsigned char)(bits>>(8*i));
  sha_update(s, length, 8);
  for (int i=0; i<8; ++i) snprintf(out+8*i, 9, "%08x", s->h[i]);
  out[64]=0;
}
static int sha_file(const char* path, char out[65]) {
  FILE* f=fopen(path,"rb"); if (!f) return 0;
  Sha256 s; sha_init(&s); unsigned char buffer[16384]; size_t n;
  while ((n=fread(buffer,1,sizeof(buffer),f)) != 0) sha_update(&s,buffer,n);
  int read_ok=!ferror(f);
  int close_ok=fclose(f)==0;
  if (!read_ok || !close_ok) return 0;
  sha_final(&s,out); return 1;
}

typedef struct {
  FILE* stream;
  int attempts;
  int returned;
  int internal_upper_bound;
  int last_error;
} Run;

static int flush_record(Run* run) {
  if (fputc('\n',run->stream)==EOF || fflush(run->stream)!=0
      || fsync(fileno(run->stream))!=0) return 0;
  return 1;
}
static void json_vec(FILE* f, const mjtNum* vec, int n) {
  fputc('[',f);
  for (int i=0; i<n; ++i) fprintf(f,"%s%.17g",i?",":"",(double)vec[i]);
  fputc(']',f);
}
static void json_hex_vec(FILE* f, const mjtNum* vec, int n) {
  fputc('[',f);
  for (int i=0; i<n; ++i) fprintf(f,"%s\"%a\"",i?",":"",(double)vec[i]);
  fputc(']',f);
}
static int finite_vec(const mjtNum* vec, int n) {
  for (int i=0; i<n; ++i) if (!isfinite((double)vec[i])) return 0;
  return 1;
}
static int bits_equal(const mjtNum* a, const double* b, int n) {
  if (sizeof(mjtNum)!=sizeof(double)) return 0;
  return memcmp(a,b,(size_t)n*sizeof(double))==0;
}
static int same_input(const mjModel* m, const mjData* d, const KernelPose* p,
                      const mjOption* options) {
  return bits_equal(d->geom_xpos,p->wheel_pos,3)
      && bits_equal(d->geom_xmat,p->wheel_mat,9)
      && bits_equal(d->geom_xpos+3,p->box_pos,3)
      && bits_equal(d->geom_xmat+9,p->box_mat,9)
      && memcmp(&m->opt,options,sizeof(*options))==0;
}
static void install_pose(mjData* d, const KernelPose* p) {
  memcpy(d->geom_xpos,p->wheel_pos,3*sizeof(double));
  memcpy(d->geom_xmat,p->wheel_mat,9*sizeof(double));
  memcpy(d->geom_xpos+3,p->box_pos,3*sizeof(double));
  memcpy(d->geom_xmat+9,p->box_mat,9*sizeof(double));
}

/* Source-equivalent arithmetic to mju_makeFrame; the raw kernel normal is retained. */
static int comparison_frame(const mjPreContact* con, mjtNum frame[9]) {
  memcpy(frame,con->normal,3*sizeof(mjtNum));
  memset(frame+3,0,6*sizeof(mjtNum));
  if (mju_normalize3(frame)<0.5) return 0;
  if (frame[1]<0.5 && frame[1]>-0.5) frame[4]=1;
  else frame[5]=1;
  mjtNum dot=mju_dot3(frame,frame+3);
  for (int i=0; i<3; ++i) frame[3+i]-=frame[i]*dot;
  if (mju_normalize3(frame+3)<0.5) return 0;
  mju_cross(frame+6,frame,frame+3);
  return finite_vec(frame,9);
}
static int box_support_valid(const KernelPose* p, const mjPreContact* con,
                             const mjtNum frame[9], double* residual_out) {
  double tolerance=fabs((double)con->dist)+fmax(kPairMargin,fmax(kWheelMargin,kBoxMargin))+1e-7;
  double residual_squared=0; int found=0, inside=1;
  for (int i=0; i<3; ++i) {
    double normal=-(double)frame[i];  /* box -> wheel; kernel order is wheel -> box */
    double center=p->box_pos[i], half=kBoxSize[i], position=con->pos[i];
    double support=center+copysign(half,normal);
    int candidate=fabs(normal)>1e-6 && fabs(position-support)<=tolerance;
    if (candidate) found=1; else residual_squared+=normal*normal;
    if (fabs(position-center)>half+tolerance) inside=0;
  }
  *residual_out=sqrt(residual_squared);
  return found && inside && *residual_out<=0.0021;
}

typedef struct {
  int finite;
  int inputs_unchanged;
  int count;
  int all_geometry_valid;
  int has_bad_downward;
  int bad_matches_reference;
  int has_good;
  int first_support_valid;
  int invalid_count;
  int bitwise_reference_equal;
  int reference_correspondence;
} PairResult;

static int pair_call(Run* run, const mjModel* m, mjData* d, mjfCollision kernel,
                     int ordinal, const KernelPose* pose, int multiccd_off,
                     PairResult* result) {
  memset(result,0,sizeof(*result));
  install_pose(d,pose);
  mjOption options=m->opt;
  if (!same_input(m,d,pose,&options)) return 0;
  ++run->attempts; run->internal_upper_bound+=multiccd_off?1:5;
  fprintf(run->stream,"{\"event\":\"pair_attempt\",\"ordinal\":%d,\"native_index\":%d,"
          "\"multiccd_off\":%s,\"attempted_pair_calls\":%d,\"internal_penetration_upper_bound\":%d}",
          ordinal,pose->native_index,multiccd_off?"true":"false",run->attempts,run->internal_upper_bound);
  if (!flush_record(run)) return 0;  /* no kernel entry without a durable attempt */

  mjPreContact contacts[mjMAXCONPAIR]; memset(contacts,0,sizeof(contacts));
  int n=kernel(m,d,contacts,0,1,kPairMargin);
  ++run->returned;
  result->count=n;
  result->inputs_unchanged=same_input(m,d,pose,&options);
  result->finite=(n>=0 && n<=MAX_PAIR_CONTACTS);
  for (int i=0; result->finite && i<n; ++i) {
    result->finite=isfinite((double)contacts[i].dist)
      && finite_vec(contacts[i].pos,3) && finite_vec(contacts[i].normal,3)
      && finite_vec(contacts[i].tangent,3);
  }
  fprintf(run->stream,"{\"event\":\"pair_return\",\"ordinal\":%d,\"native_index\":%d,"
          "\"multiccd_off\":%s,\"returned_count\":%d,\"finite\":%s,\"inputs_unchanged\":%s,"
          "\"original_score_overridden\":false,\"qualification_granted\":false,\"rl_gate_open\":false",
          ordinal,pose->native_index,multiccd_off?"true":"false",n,
          result->finite?"true":"false",result->inputs_unchanged?"true":"false");
  if (!result->finite) {
    fprintf(run->stream,"}");
    return flush_record(run) && result->inputs_unchanged;
  }
  result->all_geometry_valid=1;
  result->bitwise_reference_equal=!multiccd_off && n==pose->nref;
  result->reference_correspondence=!multiccd_off && n==pose->nref;
  fprintf(run->stream,",\"candidates\":[");
  for (int i=0; i<n; ++i) {
    mjtNum frame[9];
    if (!comparison_frame(&contacts[i],frame)) {
      run->last_error=1;
      fprintf(run->stream,"],\"frame_error_candidate\":%d}",i);
      flush_record(run);
      return 0;
    }
    double residual=0;
    int valid=box_support_valid(pose,&contacts[i],frame,&residual);
    result->all_geometry_valid &= valid;
    result->has_good |= valid;
    result->has_bad_downward |= !valid && -frame[2]<-0.9;
    result->invalid_count += !valid;
    if (i==0) result->first_support_valid=valid;
    int equal=0;
    int old_index=-1;
    double reference_position_delta[3]={0,0,0};
    double reference_distance_delta=0, reference_normal_dot=0, reference_frame_max_delta=0;
    int reference_corresponds=0;
    if (i<pose->nref && !multiccd_off) {
      const KernelReference* ref=&pose->refs[i]; old_index=ref->original_index;
      equal=bits_equal(contacts[i].pos,ref->pos,3)
          && memcmp(&contacts[i].dist,&ref->dist,sizeof(double))==0
          && bits_equal(frame,ref->frame,9);
      result->bitwise_reference_equal &= equal;
      double squared=0;
      for (int axis=0; axis<3; ++axis) {
        reference_position_delta[axis]=(double)contacts[i].pos[axis]-ref->pos[axis];
        squared+=reference_position_delta[axis]*reference_position_delta[axis];
        reference_normal_dot+=(double)frame[axis]*ref->frame[axis];
      }
      reference_distance_delta=(double)contacts[i].dist-ref->dist;
      for (int axis=0; axis<9; ++axis) {
        double delta=fabs((double)frame[axis]-ref->frame[axis]);
        if (delta>reference_frame_max_delta) reference_frame_max_delta=delta;
      }
      double position_limit=fabs(ref->dist)+kPairMargin+1e-7;
      reference_corresponds=sqrt(squared)<=position_limit
        && fabs(reference_distance_delta)<=position_limit && reference_normal_dot>0.99;
      result->reference_correspondence &= reference_corresponds;
      if (pose->native_index==3486 && old_index==5 && !valid && -frame[2]<-0.9) {
        result->bad_matches_reference=reference_corresponds;
      }
    }
    fprintf(run->stream,"%s{\"candidate_index\":%d,\"old_contact_index_same_order\":%d,"
            "\"distance_m\":%.17g,\"distance_hex\":\"%a\",\"position_world_m\":",
            i?",":"",i,old_index,(double)contacts[i].dist,(double)contacts[i].dist);
    json_vec(run->stream,contacts[i].pos,3);
    fprintf(run->stream,",\"position_hex\":"); json_hex_vec(run->stream,contacts[i].pos,3);
    fprintf(run->stream,",\"kernel_normal_geom0_to_geom1\":");
    json_vec(run->stream,contacts[i].normal,3);
    fprintf(run->stream,",\"kernel_normal_hex\":"); json_hex_vec(run->stream,contacts[i].normal,3);
    fprintf(run->stream,",\"kernel_tangent\":"); json_vec(run->stream,contacts[i].tangent,3);
    fprintf(run->stream,",\"kernel_tangent_hex\":"); json_hex_vec(run->stream,contacts[i].tangent,3);
    fprintf(run->stream,",\"local_driver_comparison_frame\":"); json_vec(run->stream,frame,9);
    fprintf(run->stream,",\"local_driver_comparison_frame_hex\":"); json_hex_vec(run->stream,frame,9);
    fprintf(run->stream,",\"saved_reference_bitwise_equal\":%s,"
            "\"saved_reference_correspondence\":%s,\"reference_position_delta_m\":[%.17g,%.17g,%.17g],"
            "\"reference_distance_delta_m\":%.17g,\"reference_normal_dot\":%.17g,"
            "\"reference_frame_max_abs_delta\":%.17g,\"box_support_valid\":%s,"
            "\"normal_cone_residual\":%.17g}",equal?"true":"false",
            reference_corresponds?"true":"false",reference_position_delta[0],
            reference_position_delta[1],reference_position_delta[2],reference_distance_delta,
            reference_normal_dot,reference_frame_max_delta,valid?"true":"false",residual);
  }
  fprintf(run->stream,"],\"bitwise_reference_equal\":%s,\"reference_correspondence\":%s,"
          "\"all_geometry_valid\":%s,"
          "\"has_bad_downward\":%s,\"bad_matches_reference\":%s,"
          "\"first_support_valid\":%s,\"invalid_count\":%d}",
          result->bitwise_reference_equal?"true":"false",
          result->reference_correspondence?"true":"false",
          result->all_geometry_valid?"true":"false",result->has_bad_downward?"true":"false",
          result->bad_matches_reference?"true":"false",
          result->first_support_valid?"true":"false",result->invalid_count);
  return flush_record(run) && result->inputs_unchanged;
}

static int verify_model(const mjModel* m) {
  if (sizeof(mjtNum)!=sizeof(double) || m->ngeom!=2 || m->nbody!=1 || m->njnt!=0
      || m->nq!=0 || m->nv!=0 || m->nu!=0 || m->nsensor!=0 || m->nplugin!=0) return 0;
  if (m->geom_type[0]!=mjGEOM_CYLINDER || m->geom_type[1]!=mjGEOM_BOX) return 0;
  if (!bits_equal(m->geom_size,kWheelSize,3) || !bits_equal(m->geom_size+3,kBoxSize,3)) return 0;
  if (memcmp(&m->geom_margin[0],&kWheelMargin,sizeof(double))!=0
      || memcmp(&m->geom_margin[1],&kBoxMargin,sizeof(double))!=0) return 0;
  if (!isfinite((double)m->geom_rbound[0]) || !isfinite((double)m->geom_rbound[1])
      || m->geom_rbound[0]<=0 || m->geom_rbound[1]<=0) return 0;
  if (m->opt.disableflags!=0 || m->opt.enableflags!=0 || m->opt.ccd_iterations!=35
      || m->opt.ccd_tolerance!=1e-6) return 0;
  return mjCOLLISIONFUNC[mjGEOM_CYLINDER][mjGEOM_BOX]!=NULL;
}

int main(int argc, char** argv) {
  /* BLOCKED by the zero-dynamic-call contract: MuJoCo XML model compilation
   * can enter an internal mj_step validation path. No certified allocation
   * route is available yet. Keep the prepared pair code for static review,
   * but make this translation unit incapable of entering the engine. */
  (void)argc; (void)argv;
  fprintf(stderr,"BLOCKED: model construction may execute internal mj_step; no kernel calls made\n");
  return 3;

  if (argc!=3) {
    fprintf(stderr,"usage: %s GENERATED_XML EXISTING_EMPTY_OUTPUT_DIR\n",argv[0]);
    return 2;
  }
  char log_path[4096];
  if (snprintf(log_path,sizeof(log_path),"%s/kernel_events.ndjson",argv[2])
      >=(int)sizeof(log_path)) return 2;
  int fd=open(log_path,O_WRONLY|O_CREAT|O_EXCL,0644);
  if (fd<0) { perror("exclusive event log"); return 2; }
  Run run={0}; run.stream=fdopen(fd,"w");
  if (!run.stream) { close(fd); return 2; }
  int status=1, model_loads=0, data_allocations=0;
  mjModel* m=NULL; mjData* d=NULL;
  char loaded_path[4096],loaded_sha[65];
  Dl_info info={0};
  if (!dladdr((void*)(uintptr_t)&mj_version,&info) || !info.dli_fname
      || !realpath(info.dli_fname,loaded_path) || !sha_file(loaded_path,loaded_sha)) {
    fprintf(stderr,"unable to identify loaded MuJoCo library\n"); goto cleanup;
  }
  if (strcmp(loaded_path,KERNEL_EXPECTED_LIB_PATH)!=0
      || strcmp(loaded_sha,KERNEL_EXPECTED_LIB_SHA256)!=0) {
    fprintf(stderr,"loaded MuJoCo library identity differs from pinned input\n"); goto cleanup;
  }
  int version=mj_version(); const char* version_string=mj_versionString();
  if (version!=mjVERSION_HEADER || version!=3012000 || strcmp(version_string,"3.12.0")!=0) {
    fprintf(stderr,"MuJoCo header/runtime version mismatch\n"); goto cleanup;
  }
  fprintf(run.stream,"{\"event\":\"library_verified\",\"path\":\"%s\",\"sha256\":\"%s\","
          "\"header_version\":%d,\"runtime_version\":%d,\"runtime_version_string\":\"%s\","
          "\"fixture_sha256\":\"%s\"}",loaded_path,loaded_sha,mjVERSION_HEADER,version,version_string,
          KERNEL_FIXTURE_SHA256);
  if (!flush_record(&run)) goto cleanup;

  /* No model constructor is allowed here until a separate source-backed
   * contract proves that it performs zero integration/step calls. */
  fprintf(stderr,"BLOCKED: no certified model constructor\n"); goto cleanup;
  if (!verify_model(m)) { fprintf(stderr,"compiled two-primitive metadata mismatch\n"); goto cleanup; }
  ++data_allocations;
  d=mj_makeData(m);
  if (!d) { fprintf(stderr,"mj_makeData failed\n"); goto cleanup; }
  fprintf(run.stream,"{\"event\":\"model_verified\",\"model_loads\":%d,\"data_allocations\":%d,"
          "\"ngeom\":%ld,\"nbody\":%ld,\"nq\":%ld,\"nv\":%ld,\"nu\":%ld,"
          "\"geom_ids\":{\"new_wheel\":0,\"new_box\":1,\"old_wheel\":59,\"old_box\":1},"
          "\"wheel_rbound_m\":%.17g,\"box_rbound_m\":%.17g,"
          "\"disableflags\":%d,\"enableflags\":%d,\"ccd_iterations\":%d,"
          "\"ccd_tolerance\":%.17g,\"pair_margin_m\":%.17g,\"gap_m\":0}",
          model_loads,data_allocations,m->ngeom,m->nbody,m->nq,m->nv,m->nu,
          (double)m->geom_rbound[0],(double)m->geom_rbound[1],m->opt.disableflags,
          m->opt.enableflags,m->opt.ccd_iterations,(double)m->opt.ccd_tolerance,kPairMargin);
  if (!flush_record(&run)) goto cleanup;

  mjfCollision kernel=mjCOLLISIONFUNC[mjGEOM_CYLINDER][mjGEOM_BOX];
  mjOption baseline=m->opt;
  PairResult results[3];
  for (int i=0; i<3; ++i) {
    if (run.attempts>=4 || run.internal_upper_bound+5>16) goto cleanup;
    if (!pair_call(&run,m,d,kernel,i+1,&kPoses[i],0,&results[i])) {
      fprintf(stderr,"pair call or input integrity failed at %d\n",i+1); goto cleanup;
    }
    if (!results[i].finite) { fprintf(stderr,"nonfinite pair output at %d\n",i+1); goto cleanup; }
    if (memcmp(&m->opt,&baseline,sizeof(baseline))!=0) goto cleanup;
  }
  int controls_ok=results[0].count==kPoses[0].nref && results[2].count==kPoses[2].nref
    && results[0].all_geometry_valid && results[2].all_geometry_valid
    && results[0].reference_correspondence && results[2].reference_correspondence;
  int target_bad=results[1].count==kPoses[1].nref && results[1].has_bad_downward
    && results[1].bad_matches_reference && results[1].first_support_valid
    && results[1].invalid_count==1;
  int fourth_allowed=controls_ok && target_bad;
  fprintf(run.stream,"{\"event\":\"fourth_call_gate\",\"allowed\":%s,"
          "\"controls_match_and_valid\":%s,\"target_bad_downward_reproduced\":%s,"
          "\"default_bitwise_equal\":[%s,%s,%s]}",fourth_allowed?"true":"false",
          controls_ok?"true":"false",target_bad?"true":"false",
          results[0].bitwise_reference_equal?"true":"false",
          results[1].bitwise_reference_equal?"true":"false",
          results[2].bitwise_reference_equal?"true":"false");
  if (!flush_record(&run)) goto cleanup;
  if (fourth_allowed) {
    if (run.attempts!=3 || run.internal_upper_bound+1>16) goto cleanup;
    m->opt.disableflags=baseline.disableflags|mjDSBL_MULTICCD;
    if (m->opt.disableflags!=(baseline.disableflags|mjDSBL_MULTICCD)
        || (m->opt.disableflags & mjDSBL_NATIVECCD)
        || m->opt.ccd_iterations!=baseline.ccd_iterations
        || m->opt.ccd_tolerance!=baseline.ccd_tolerance) goto cleanup;
    PairResult contrast;
    int call_ok=pair_call(&run,m,d,kernel,4,&kPoses[1],1,&contrast);
    m->opt=baseline;
    if (!call_ok || !contrast.finite || memcmp(&m->opt,&baseline,sizeof(baseline))!=0) {
      fprintf(stderr,"multiccd-off diagnostic failed\n"); goto cleanup;
    }
  }
  fprintf(run.stream,"{\"event\":\"conclusion\",\"classification\":\"%s\","
          "\"pair_attempts\":%d,\"pair_returns\":%d,\"internal_penetration_upper_bound\":%d,"
          "\"control_intervals\":0,\"native_steps\":0,\"explicit_forward_calls\":0,"
          "\"original_score_overridden\":false,\"qualification_granted\":false,\"rl_gate_open\":false}",
          target_bad?(results[0].bitwise_reference_equal && results[1].bitwise_reference_equal
                     && results[2].bitwise_reference_equal?"bitwise_reproduced":"same_geometry_anomaly"):
                     "reproduction_gap",
          run.attempts,run.returned,run.internal_upper_bound);
  if (!flush_record(&run)) goto cleanup;
  status=0;

cleanup:
  if (d) mj_deleteData(d);
  if (m) mj_deleteModel(m);
  fprintf(run.stream,"{\"event\":\"cleanup\",\"exit_status\":%d,\"model_loads\":%d,"
          "\"data_allocations\":%d,\"model_deleted\":%s,\"data_deleted\":%s,"
          "\"pair_attempts\":%d,\"pair_returns\":%d,\"internal_penetration_upper_bound\":%d,"
          "\"original_score_overridden\":false,\"qualification_granted\":false,\"rl_gate_open\":false}",
          status,model_loads,data_allocations,m?"true":"false",d?"true":"false",
          run.attempts,run.returned,run.internal_upper_bound);
  if (!flush_record(&run)) status=1;
  fclose(run.stream);
  return status;
}
