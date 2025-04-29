opencl_dep = declare_dependency(
  include_directories : util_incdir,
  dependencies : [
    core_dep,
  ],
  compile_args : [ '-DCL_TARGET_OPENCL_VERSION=120' ], # TODO: ?
)
opencl = static_library(
  'OpenCL',
  [
    'Device.cpp',
    'Error.cpp',
    'Event.cpp',
    'GetError.cpp',
    'OpenCLLibLazy.cpp',
    'Program.cpp',
  ],
  dependencies : opencl_dep,
  implicit_include_directories : false,
)
opencl_dep = declare_dependency(link_with : opencl, dependencies : [ opencl_dep ])
