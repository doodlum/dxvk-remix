dxbc_src = files([
  'dxbc_analysis.cpp',
  'dxbc_chunk_isgn.cpp',
  'dxbc_chunk_shex.cpp',
  'dxbc_common.cpp',
  'dxbc_compiler.cpp',
  'dxbc_defs.cpp',
  'dxbc_decoder.cpp',
  'dxbc_header.cpp',
  'dxbc_module.cpp',
  'dxbc_names.cpp',
  'dxbc_options.cpp',
  'dxbc_reader.cpp',
  'dxbc_util.cpp',
])

# rtx_materials.h reaches these translation units through the dxvk headers and
# includes rtx/... from src/dxvk/shaders, so the shader include path is required.
dxbc_lib = static_library('dxbc', dxbc_src,
  include_directories : [ dxvk_include_path, dxvk_shader_include_path, usd_include_paths ],
  override_options    : ['cpp_std='+dxvk_cpp_std])

dxbc_dep = declare_dependency(
  link_with           : [ dxbc_lib ],
  include_directories : [ dxvk_include_path ])
