/*
* Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/
#ifndef SHARED_CONSTANTS_H
#define SHARED_CONSTANTS_H

// contains constants shared between shader and host code

// NV-DXVK start: Shared host/shader material record stride.
#define SURFACE_MATERIAL_GPU_SIZE 112
// NV-DXVK end

static const uint8_t surfaceMaterialTypeOpaque = uint8_t(0u);
static const uint8_t surfaceMaterialTypeTranslucent = uint8_t(1u);
static const uint8_t surfaceMaterialTypeRayPortal = uint8_t(2u);
static const uint8_t surfaceMaterialTypeMask = uint8_t(0x3u);

#define COMMON_MATERIAL_FLAG_TYPE_MASK surfaceMaterialTypeMask
#define COMMON_MATERIAL_FLAG_TYPE_OFFSET(X) (2 + X)

// NOTE: Each material memory structure contains a set of flags.  The first 2 bits in that flag identify the material type (opaque, etc).
//       We must ensure all other material flags are written to byte addresses after these first two bits.  Use the COMMON_MATERIAL_FLAG_TYPE_OFFSET(x) 
//       macro, and ensure there is enough storage in the flags to represent desired bits accordingly.

// maximum value for thin film thickness in nanometers
#define OPAQUE_SURFACE_MATERIAL_THIN_FILM_MAX_THICKNESS (1500.0f)
// bits for flags field in OpaqueSurfaceMaterial
#define OPAQUE_SURFACE_MATERIAL_FLAG_USE_THIN_FILM_LAYER (1 << COMMON_MATERIAL_FLAG_TYPE_OFFSET(0))
#define OPAQUE_SURFACE_MATERIAL_FLAG_NATIVE_LANDSCAPE (1 << COMMON_MATERIAL_FLAG_TYPE_OFFSET(12))
#define OPAQUE_SURFACE_MATERIAL_FLAG_NATIVE_EFFECT (1 << COMMON_MATERIAL_FLAG_TYPE_OFFSET(13))
// A host-imported albedo texture whose format claims linear but whose contents
// are gamma encoded, so the hardware performs no conversion and the shader must.
// NV-DXVK start: Native colour metadata shares the packed 16-bit opaque flags.
#define OPAQUE_SURFACE_MATERIAL_FLAG_NATIVE_SRGB_ALBEDO (1 << COMMON_MATERIAL_FLAG_TYPE_OFFSET(8))
#ifdef __cplusplus
static_assert((OPAQUE_SURFACE_MATERIAL_FLAG_NATIVE_SRGB_ALBEDO & 0xffffu) ==
              OPAQUE_SURFACE_MATERIAL_FLAG_NATIVE_SRGB_ALBEDO);
#endif
// NV-DXVK end
#define OPAQUE_SURFACE_MATERIAL_FLAG_ALPHA_IS_THIN_FILM_THICKNESS (1 << COMMON_MATERIAL_FLAG_TYPE_OFFSET(1))
#define OPAQUE_SURFACE_MATERIAL_FLAG_IGNORE_ALPHA_CHANNEL (1 << COMMON_MATERIAL_FLAG_TYPE_OFFSET(2))
#define OPAQUE_SURFACE_MATERIAL_FLAG_IS_RAYTRACED_RENDER_TARGET (1 << COMMON_MATERIAL_FLAG_TYPE_OFFSET(3))
#define OPAQUE_SURFACE_MATERIAL_FLAG_HAS_DISPLACEMENT (1 << COMMON_MATERIAL_FLAG_TYPE_OFFSET(4))
#define OPAQUE_SURFACE_MATERIAL_FLAG_IS_HAIR_CARD (1 << COMMON_MATERIAL_FLAG_TYPE_OFFSET(5))
#define OPAQUE_SURFACE_MATERIAL_FLAG_NATIVE_RGB_NORMAL (1 << COMMON_MATERIAL_FLAG_TYPE_OFFSET(6))
// NV-DXVK start: Native foliage colour inputs.
#define OPAQUE_SURFACE_MATERIAL_FLAG_NATIVE_FOLIAGE (1 << COMMON_MATERIAL_FLAG_TYPE_OFFSET(7))
#define NATIVE_FOLIAGE_GRASS 1u
#define NATIVE_FOLIAGE_SOFT 2u
#define NATIVE_FOLIAGE_BACK 4u
#define NATIVE_FOLIAGE_NO_VERTEX_COLOR 8u
#define NATIVE_FOLIAGE_OVERRIDE_COMPLEX 16u
#define NATIVE_FOLIAGE_GAMMA_COLOR 32u
#define NATIVE_FOLIAGE_SPHERE_NORMAL 64u
// NV-DXVK end


#define OPAQUE_SURFACE_MATERIAL_INTERACTION_FLAG_HAS_HEIGHT_TEXTURE (1 << 0)
#define OPAQUE_SURFACE_MATERIAL_INTERACTION_FLAG_USE_THIN_FILM_LAYER (1 << 1)
#define OPAQUE_SURFACE_MATERIAL_INTERACTION_FLAG_NATIVE_FOLIAGE (1 << 2)
#define OPAQUE_SURFACE_MATERIAL_INTERACTION_FLAG_GRASS_LIGHTING_PROXY (1 << 2)
// flags overlap with type field when in gbuffer, which occupies last 2 bits.
#define OPAQUE_SURFACE_MATERIAL_INTERACTION_FLAG_MASK 0x3F


// Note: Bits for flags field in TranslucentSurfaceMaterial and TranslucentSurfaceMaterialInteraction
// If set, then the texture bound to transmittanceOrDiffuseTextureIndex is an albedo map for the diffuse layer
#define TRANSLUCENT_SURFACE_MATERIAL_FLAG_USE_DIFFUSE_LAYER (1 << COMMON_MATERIAL_FLAG_TYPE_OFFSET(0))
#define TRANSLUCENT_SURFACE_MATERIAL_FLAG_NATIVE_WATER (1 << COMMON_MATERIAL_FLAG_TYPE_OFFSET(1))

#endif // ifndef SHARED_CONSTANTS_H
