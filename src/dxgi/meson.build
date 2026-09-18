dxgi_res = wrc_generator.process('version.rc')

# A forwarder only. The implementation is compiled into dxvk_d3d11.dll, because
# the RTX runtime keeps global state that must exist once in the process; see
# the note on dxgi_common_src in ../d3d11/meson.build.
dxgi_dll = shared_library('dxvk_dxgi'+dll_ext, ['dxgi_forward.cpp'], dxgi_res,
  name_prefix         : '',
  install             : true,
  override_options    : ['cpp_std='+dxvk_cpp_std])

dxgi_dep = declare_dependency(
  link_with           : [ dxgi_dll ],
  include_directories : [ dxvk_include_path ])
