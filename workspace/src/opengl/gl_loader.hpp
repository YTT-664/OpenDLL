#pragma once

#include <windows.h>
#include <GL/glcorearb.h>

namespace opendll::gl {

// 集中管理 OpenGL 函数指针，避免与 glcorearb.h 中的 extern 声明冲突。
// 所有 1.2+ 函数必须经 wglGetProcAddress 加载；1.1 函数从 opengl32.dll 加载。
struct Funcs {
    // buffer
    PFNGLGENBUFFERSPROC genBuffers = nullptr;
    PFNGLDELETEBUFFERSPROC deleteBuffers = nullptr;
    PFNGLBINDBUFFERPROC bindBuffer = nullptr;
    PFNGLBINDBUFFERBASEPROC bindBufferBase = nullptr;
    PFNGLBUFFERDATAPROC bufferData = nullptr;
    PFNGLBUFFERSUBDATAPROC bufferSubData = nullptr;
    PFNGLMAPBUFFERRANGEPROC mapBufferRange = nullptr;
    PFNGLUNMAPBUFFERPROC unmapBuffer = nullptr;
    // shader
    PFNGLCREATESHADERPROC createShader = nullptr;
    PFNGLSHADERSOURCEPROC shaderSource = nullptr;
    PFNGLCOMPILESHADERPROC compileShader = nullptr;
    PFNGLGETSHADERIVPROC getShaderiv = nullptr;
    PFNGLGETSHADERINFOLOGPROC getShaderInfoLog = nullptr;
    PFNGLDELETESHADERPROC deleteShader = nullptr;
    // program
    PFNGLCREATEPROGRAMPROC createProgram = nullptr;
    PFNGLATTACHSHADERPROC attachShader = nullptr;
    PFNGLLINKPROGRAMPROC linkProgram = nullptr;
    PFNGLGETPROGRAMIVPROC getProgramiv = nullptr;
    PFNGLGETPROGRAMINFOLOGPROC getProgramInfoLog = nullptr;
    PFNGLDELETEPROGRAMPROC deleteProgram = nullptr;
    PFNGLUSEPROGRAMPROC useProgram = nullptr;
    PFNGLSHADERSTORAGEBLOCKBINDINGPROC shaderStorageBlockBinding = nullptr;
    // compute
    PFNGLDISPATCHCOMPUTEPROC dispatchCompute = nullptr;
    PFNGLMEMORYBARRIERPROC memoryBarrier = nullptr;
    // 1.1
    PFNGLGETSTRINGPROC getString = nullptr;
    PFNGLGETINTEGERVPROC getIntegerv = nullptr;
    PFNGLGETERRORPROC getError = nullptr;
    PFNGLFINISHPROC finish = nullptr;
};

// 加载全部函数指针（必须在上下文 make current 之后调用）。
// 关键函数缺失时返回 false。
bool load(Funcs& f);

}  // namespace opendll::gl
