ct_dep = declare_dependency(
  include_directories : util_incdir,
  dependencies : [
    core_dep,
    hdf5_dep,
  ],
)
ct = static_library(
  'CT',
  [
    'CTFloat.cpp',
    'DataFiles.cpp',
  ],
  dependencies : ct_dep,
  implicit_include_directories : false,
)
ct_dep = declare_dependency(link_with : ct, dependencies : [ ct_dep ])
