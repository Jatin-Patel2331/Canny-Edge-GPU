project(
    'CannyEdgeDetection',
    ['cpp', 'c'],
    default_options : [
        'buildtype=debugoptimized',
        'optimization=3',
        'cpp_std=c++14',
        'warning_level=3',
    ],
)

subdir('lib')

# Add OpenCV dependency
opencv_dep = dependency('opencv4', required: true)

executable('CannyEdgeDetection',
    [
        'src/main.cpp',
        xxd.process('src/canny.cl'),
    ],
    extra_files: [
        'src/canny.cl',
    ],
    dependencies : [
        meson.get_compiler('c').find_library('glut'),
        meson.get_compiler('c').find_library('GLEW'),
        meson.get_compiler('c').find_library('GL'),
        core_dep,
        opencl_dep,
        ct_dep,
        opencv_dep,
    ],
)

