boost_dep = dependency('boost', modules: ['system', 'filesystem'])

libhdf5_dep = dependency('hdf5-serial')

eigen3_dep = dependency('eigen3')

hdf5_dep = declare_dependency(
  include_directories : util_incdir,
  dependencies : [
    core_dep,
    boost_dep,
    libhdf5_dep,
    eigen3_dep,
  ],
)
hdf5 = static_library(
  'HDF5',
  [
    'Array.cpp',
    'AtomicType.cpp',
    'AtomicTypes.cpp',
    'Attribute.cpp',
    'BaseTypes.cpp',
    'ComplexConversion.cpp',
    'CompoundType.cpp',
    'DataSet.cpp',
    'DataSpace.cpp',
    'DataType.cpp',
    'DataTypes.cpp',
    'DelayedArray.cpp',
    'Exception.cpp',
    'File.cpp',
    'Group.cpp',
    'IdComponent.cpp',
    'Matlab.cpp',
    'MatlabDiagMatrix3.cpp',
    'MatlabVector2.cpp',
    'MatlabVector3.cpp',
    'Object.cpp',
    'OpaqueType.cpp',
    'PropList.cpp',
    'PropLists.cpp',
    'ReferenceType.cpp',
    'SerializationKey.cpp',
    'Type.cpp',
    'Util.cpp',
  ],
  dependencies : hdf5_dep,
  implicit_include_directories : false,
)
hdf5_dep = declare_dependency(link_with : hdf5, dependencies : [ hdf5_dep ])
