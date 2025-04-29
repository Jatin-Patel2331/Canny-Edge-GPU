dl_dep = meson.get_compiler('c').find_library('dl', required : false)

boost_dep = dependency('boost', modules: ['system', 'filesystem', 'program_options', 'iostreams'])

core_dep = declare_dependency(
  include_directories : util_incdir,
  dependencies : [
    dl_dep,
    boost_dep,
  ],
)
core = static_library(
  'Core',
  [
    'Assert.cpp',
    'CheckedCast.cpp',
    'Error.cpp',
    'Exception.cpp',
    'Image.cpp',
    'Memory.cpp',
    'NumericException.cpp',
    'OStream.cpp',
    'StaticCache.cpp',
    'StrError.c',
    'StringUtil.cpp',
    'Time.cpp',
    'TimeSpan.cpp',
    'Type.cpp',
    'WindowsError.cpp',
  ],
  implicit_include_directories : false,
  dependencies : core_dep,
  include_directories : [ include_directories('../lib/fmt/include') ],
)
core_dep = declare_dependency(link_with : core, dependencies : [ core_dep ])
