/* Generated only from fixed JSON; every floating literal is binary64 hex. */
#ifndef D1_CONTACT_KERNEL_INPUT_H
#define D1_CONTACT_KERNEL_INPUT_H
#define KERNEL_POSE_COUNT 3
#define KERNEL_MAX_REFERENCE 5
#define KERNEL_EXPECTED_LIB_PATH "/home/lyh/.local/lib/python3.10/site-packages/mujoco/libmujoco.so.3.12.0"
#define KERNEL_EXPECTED_LIB_SHA256 "bd3f702ace8a31e1046f746880387858a981d11b01772a55ebef48ffd55ea5b8"
#define KERNEL_FIXTURE_SHA256 "91cb3c9d1137b11c39f071de01ddb9e2f9ba83ca90af8e04e3cc565800852a55"
typedef struct { int original_index; double pos[3]; double dist; double frame[9]; } KernelReference;
typedef struct { int native_index; double wheel_pos[3]; double wheel_mat[9];
  double box_pos[3]; double box_mat[9]; int nref;
  KernelReference refs[KERNEL_MAX_REFERENCE]; } KernelPose;
static const double kWheelSize[3] = {0x1.645a1cac08312p-4, 0x1.47ae147ae147bp-6, 0x0.0p+0};
static const double kBoxSize[3] = {0x1.70a3d70a3d70ap-3, 0x1.3d70a3d70a3d7p-1, 0x1.eb851eb851eb8p-8};
static const double kWheelMargin = 0x1.0624dd2f1a9fcp-10;
static const double kBoxMargin = 0x0.0p+0;
static const double kPairMargin = 0x1.0624dd2f1a9fcp-10;
static const KernelPose kPoses[KERNEL_POSE_COUNT] = {
  {
    3485, {-0x1.842ac4606d195p+1, 0x1.b7ecdb6e68ee8p-3, 0x1.a5b78db755a94p-4},
    {0x1.d6f7554d046b8p-1, 0x1.91abb20e818a0p-2, -0x1.e05a32c1a0200p-12, 0x1.90ba7c7287840p-8, -0x1.af98d88efc914p-7, 0x1.fff22df8e131bp-1, 0x1.919f458226728p-2, -0x1.d6eafcd0b68d7p-1, -0x1.db99e8f6a3c6cp-7},
    {-0x1.8ccccccccccccp+1, 0x0.0p+0, 0x1.eb851eb851eb8p-8},
    {0x1.0000000000000p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x1.0000000000000p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x1.0000000000000p+0},
    3, {
      {4, {-0x1.842b5047ca59cp+1, 0x1.de4b78034204cp-3, 0x1.f69d3a2fef147p-7}, 0x1.63036f2e3ca78p-11, {-0x1.36f95a1b525d8p-29, -0x1.b38d7ccdb7ae1p-41, -0x1.0000000000000p+0, -0x1.088ac986dd39bp-69, 0x1.0000000000000p+0, -0x1.b38d7ccdb7ae1p-41, 0x1.0000000000000p+0, 0x0.0p+0, -0x1.36f95a1b525d8p-29}},
      {5, {-0x1.8431091221724p+1, 0x1.de4b6db26f6bap-3, 0x1.f69e06f7acef4p-7}, 0x1.63036f2e3ca78p-11, {-0x1.062500781cd92p-10, -0x1.b356c54831118p-41, -0x1.ffffef390394bp-1, -0x1.bdc9ba3ea7623p-51, 0x1.0000000000000p+0, -0x1.b356b70449b90p-41, 0x1.ffffef390394bp-1, 0x0.0p+0, -0x1.062500781cd92p-10}},
      {6, {-0x1.8425977e98d1ep+1, 0x1.de4b83017bc0ap-3, 0x1.f69de4392b516p-7}, 0x1.63036f2e3ca78p-11, {0x1.0624b2c5d6b31p-10, -0x1.0a0389d6d5d04p-40, -0x1.ffffef390d86ap-1, 0x1.1065c1554ab2bp-50, 0x1.0000000000000p+0, -0x1.0a03811f5c288p-40, 0x1.ffffef390d86ap-1, 0x0.0p+0, 0x1.0624b2c5d6b31p-10}},
    }
  },
  {
    3486, {-0x1.841e8441b4020p+1, 0x1.b7ed1f0e37046p-3, 0x1.a5b6eb794749ep-4},
    {0x1.d7d268da16b68p-1, 0x1.8da21560a7ea7p-2, -0x1.df1288b248c00p-12, 0x1.8cf3404e5ae00p-8, -0x1.b07631cee6cf4p-7, 0x1.fff22e124fa73p-1, 0x1.8d95c5139f89bp-2, -0x1.d7c60968fc445p-1, -0x1.db9885963bdcap-7},
    {-0x1.8ccccccccccccp+1, 0x0.0p+0, 0x1.eb851eb851eb8p-8},
    {0x1.0000000000000p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x1.0000000000000p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x1.0000000000000p+0},
    3, {
      {4, {-0x1.841f1157f84b0p+1, 0x1.de4bbd8c2a084p-3, 0x1.f69ab491e8393p-7}, 0x1.62b2bb6fa57bap-11, {-0x1.4248923c8fd93p-29, -0x1.09ad22d166652p-40, -0x1.0000000000000p+0, -0x1.4e771640c568bp-69, 0x1.0000000000000p+0, -0x1.09ad22d166652p-40, 0x1.0000000000000p+0, 0x0.0p+0, -0x1.4248923c8fd93p-29}},
      {5, {-0x1.83cbc0d3199fdp+1, 0x1.c5605c9f1e8b7p-3, 0x1.00827750ef60ap-6}, 0x1.62b2bb6fa57bap-11, {-0x1.10a316788ed5ap-8, -0x1.0a83c329e5941p-19, 0x1.fffedda4b1afap-1, -0x1.1bd5c8a69f3e5p-27, 0x1.fffffffffbaa3p-1, 0x1.0a832c05bf1c0p-19, -0x1.fffedda4b6058p-1, 0x1.0000000000000p-78, -0x1.10a316789124ap-8}},
      {6, {-0x1.84195891f60ebp+1, 0x1.de4bc882e2077p-3, 0x1.f69b5e40767f9p-7}, 0x1.62b2bb6fa57bap-11, {0x1.0624b14ab012ap-10, -0x1.e2c331f742e14p-41, -0x1.ffffef390db73p-1, 0x1.ee58f6ebe339ap-51, 0x1.0000000000000p+0, -0x1.e2c3222593625p-41, 0x1.ffffef390db73p-1, 0x1.0000000000000p-103, 0x1.0624b14ab012ap-10}},
    }
  },
  {
    3487, {-0x1.8412455ea545ep+1, 0x1.b7ed65cd88d45p-3, 0x1.a5b4a460924b2p-4},
    {0x1.d8ab3645c975ep-1, 0x1.8996e2f362cb6p-2, -0x1.ddb39f004fa00p-12, 0x1.8928dffce18e0p-8, -0x1.b151782ac0274p-7, 0x1.fff22e2ef13f1p-1, 0x1.898aaf0967894p-2, -0x1.d89ecfeea629fp-1, -0x1.db96f0b6fd88ep-7},
    {-0x1.8ccccccccccccp+1, 0x0.0p+0, 0x1.eb851eb851eb8p-8},
    {0x1.0000000000000p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x1.0000000000000p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x1.0000000000000p+0},
    3, {
      {4, {-0x1.8412d3a4e8758p+1, 0x1.de4c0679aa35ap-3, 0x1.f6919c002b961p-7}, 0x1.618fa93a67a14p-11, {-0x1.4e31d7f58612fp-29, -0x1.07e82fa2ba910p-40, -0x1.0000000000000p+0, -0x1.5884503888fd3p-69, 0x1.0000000000000p+0, -0x1.07e82fa2ba910p-40, 0x1.0000000000000p+0, 0x0.0p+0, -0x1.4e31d7f58612fp-29}},
      {5, {-0x1.84188c65841fcp+1, 0x1.de4bfc385e2fap-3, 0x1.f692697b69a7ep-7}, 0x1.618fa93a67a14p-11, {-0x1.06250364ed471p-10, -0x1.07c77396b2224p-40, -0x1.ffffef390334ep-1, -0x1.0e1c439f43d79p-50, 0x1.0000000000000p+0, -0x1.07c76af1f2270p-40, 0x1.ffffef390334ep-1, 0x0.0p+0, -0x1.06250364ed471p-10}},
      {6, {-0x1.840d1ae574402p+1, 0x1.de4c11685b5a0p-3, 0x1.f692455271e23p-7}, 0x1.618fa93a67a14p-11, {0x1.0624afcbd779ep-10, -0x1.1fbab274a2987p-40, -0x1.ffffef390de84p-1, 0x1.26a24e7a3206ep-50, 0x1.0000000000000p+0, -0x1.1fbaa906ffc53p-40, 0x1.ffffef390de84p-1, 0x0.0p+0, 0x1.0624afcbd779ep-10}},
    }
  },
};
#define KERNEL_CONTRACT_02_SHA256 "0869d5836add5dc12a295d5ac4618bff984244c9703516202f9485a29fdf4768"
static const double kWheelRbound = 0x1.6da5995739234p-4;
static const double kBoxRbound = 0x1.4a91dba663dbbp-1;
#endif
