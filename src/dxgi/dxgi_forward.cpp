// The DXGI implementation is compiled into dxvk_d3d11.dll rather than here.
//
// The RTX runtime keeps global state -- the option singleton above all -- and
// the static library that holds it, linked into two DLLs, gives each its own
// copy. With DXGI in its own DLL the options were created there while the
// device, created in the D3D11 DLL, found its own copy still null. So this DLL
// keeps its name and its exports and forwards them; see the note on
// dxgi_common_src in ../d3d11/meson.build.
//
// Forwarders are declared to the linker rather than through the module
// definition file, which reads Module.Function as an ordinary symbol name.

#pragma comment(linker, "/EXPORT:CreateDXGIFactory=dxvk_d3d11.CreateDXGIFactory,@9")
#pragma comment(linker, "/EXPORT:CreateDXGIFactory1=dxvk_d3d11.CreateDXGIFactory1,@10")
#pragma comment(linker, "/EXPORT:CreateDXGIFactory2=dxvk_d3d11.CreateDXGIFactory2,@11")
#pragma comment(linker, "/EXPORT:DXGIDeclareAdapterRemovalSupport=dxvk_d3d11.DXGIDeclareAdapterRemovalSupport,@16")
#pragma comment(linker, "/EXPORT:DXGIGetDebugInterface1=dxvk_d3d11.DXGIGetDebugInterface1,@17")
