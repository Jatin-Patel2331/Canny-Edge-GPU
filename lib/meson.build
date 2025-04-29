# xxd_exe = find_program('xxd')
# xxd = generator(xxd_exe,
#     output : '@PLAINNAME@.xxd.c',
#     arguments : [ '-i', '-n', '@PLAINNAME@', '@INPUT@', '@OUTPUT@' ]
# )

prog_python = find_program('python3')
xxd_py = 'xxd.py'
xxd = generator(
    prog_python,
    output : '@PLAINNAME@.xxd.c',
    arguments : [
        join_paths(meson.current_source_dir(), xxd_py),
        '-i', '-n', '@PLAINNAME@', '@INPUT@', '@OUTPUT@',
    ],
)

util_incdir = include_directories('.')

subdir('Core')
# subdir('vx')
subdir('OpenCL')
#subdir('Math')
subdir('HDF5')
subdir('CT')
