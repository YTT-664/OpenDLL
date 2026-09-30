#include "gl_loader.hpp"

namespace opendll::gl {

namespace {

void* resolve(const char* name) {
    void* p = reinterpret_cast<void*>(wglGetProcAddress(name));
    if (p != nullptr) {
        return p;
    }
    // 1.1 函数不一定经 wglGetProcAddress 提供，回退到 opengl32.dll。
    static HMODULE opengl32 = GetModuleHandleA("opengl32.dll");
    if (opengl32 != nullptr) {
        p = reinterpret_cast<void*>(GetProcAddress(opengl32, name));
    }
    return p;
}

}  // namespace

bool load(Funcs& f) {
    f.genBuffers = reinterpret_cast<PFNGLGENBUFFERSPROC>(resolve("glGenBuffers"));
    f.deleteBuffers = reinterpret_cast<PFNGLDELETEBUFFERSPROC>(resolve("glDeleteBuffers"));
    f.bindBuffer = reinterpret_cast<PFNGLBINDBUFFERPROC>(resolve("glBindBuffer"));
    f.bindBufferBase = reinterpret_cast<PFNGLBINDBUFFERBASEPROC>(resolve("glBindBufferBase"));
    f.bufferData = reinterpret_cast<PFNGLBUFFERDATAPROC>(resolve("glBufferData"));
    f.bufferSubData = reinterpret_cast<PFNGLBUFFERSUBDATAPROC>(resolve("glBufferSubData"));
    f.mapBufferRange = reinterpret_cast<PFNGLMAPBUFFERRANGEPROC>(resolve("glMapBufferRange"));
    f.unmapBuffer = reinterpret_cast<PFNGLUNMAPBUFFERPROC>(resolve("glUnmapBuffer"));

    f.createShader = reinterpret_cast<PFNGLCREATESHADERPROC>(resolve("glCreateShader"));
    f.shaderSource = reinterpret_cast<PFNGLSHADERSOURCEPROC>(resolve("glShaderSource"));
    f.compileShader = reinterpret_cast<PFNGLCOMPILESHADERPROC>(resolve("glCompileShader"));
    f.getShaderiv = reinterpret_cast<PFNGLGETSHADERIVPROC>(resolve("glGetShaderiv"));
    f.getShaderInfoLog = reinterpret_cast<PFNGLGETSHADERINFOLOGPROC>(resolve("glGetShaderInfoLog"));
    f.deleteShader = reinterpret_cast<PFNGLDELETESHADERPROC>(resolve("glDeleteShader"));

    f.createProgram = reinterpret_cast<PFNGLCREATEPROGRAMPROC>(resolve("glCreateProgram"));
    f.attachShader = reinterpret_cast<PFNGLATTACHSHADERPROC>(resolve("glAttachShader"));
    f.linkProgram = reinterpret_cast<PFNGLLINKPROGRAMPROC>(resolve("glLinkProgram"));
    f.getProgramiv = reinterpret_cast<PFNGLGETPROGRAMIVPROC>(resolve("glGetProgramiv"));
    f.getProgramInfoLog = reinterpret_cast<PFNGLGETPROGRAMINFOLOGPROC>(resolve("glGetProgramInfoLog"));
    f.deleteProgram = reinterpret_cast<PFNGLDELETEPROGRAMPROC>(resolve("glDeleteProgram"));
    f.useProgram = reinterpret_cast<PFNGLUSEPROGRAMPROC>(resolve("glUseProgram"));
    f.shaderStorageBlockBinding = reinterpret_cast<PFNGLSHADERSTORAGEBLOCKBINDINGPROC>(
        resolve("glShaderStorageBlockBinding"));

    f.dispatchCompute = reinterpret_cast<PFNGLDISPATCHCOMPUTEPROC>(resolve("glDispatchCompute"));
    f.memoryBarrier = reinterpret_cast<PFNGLMEMORYBARRIERPROC>(resolve("glMemoryBarrier"));

    f.getString = reinterpret_cast<PFNGLGETSTRINGPROC>(resolve("glGetString"));
    f.getIntegerv = reinterpret_cast<PFNGLGETINTEGERVPROC>(resolve("glGetIntegerv"));
    f.getError = reinterpret_cast<PFNGLGETERRORPROC>(resolve("glGetError"));
    f.finish = reinterpret_cast<PFNGLFINISHPROC>(resolve("glFinish"));

    // 校验最小可用集（compute shader 必需）
    return f.dispatchCompute != nullptr && f.createShader != nullptr &&
           f.createProgram != nullptr && f.genBuffers != nullptr &&
           f.mapBufferRange != nullptr;
}

}  // namespace opendll::gl
