/*
 * Copyright (c) 2018 Steffen Kieß
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "OpenCLLibLazy.hpp"

#include <Core/Assert.hpp>
#include <Core/Exception.hpp>
#include <Core/OStream.hpp>
#include <Core/StaticCache.hpp>
#include <Core/Util.hpp>
#include <Core/WindowsError.hpp>

#include <boost/static_assert.hpp>
#include <memory>

#include <cstring>

#include <sstream>

#if OS_UNIX
#include <dlfcn.h>
#elif OS_WIN
#include <windows.h>
#endif

// TODO: move all the libdl-related code away

// TODO: test under windows

namespace {
#if OS_UNIX
typedef void* HandleType;
typedef void* SymbolType;

template <typename T>
inline SymbolType fromFunPtr(T funptr) {
  BOOST_STATIC_ASSERT(sizeof(SymbolType) == sizeof(T));
  SymbolType res;
  memcpy(&res, &funptr, sizeof(SymbolType));
  return res;
}
template <typename T>
inline T toFunPtr(SymbolType ptr) {
  BOOST_STATIC_ASSERT(sizeof(SymbolType) == sizeof(T));
  T res;
  memcpy(&res, &ptr, sizeof(SymbolType));
  return res;
}
#elif OS_WIN
typedef HMODULE HandleType;
typedef FARPROC SymbolType;

template <typename T>
inline SymbolType fromFunPtr(T funptr) {
  //return static_cast<SymbolType> (funptr);
  return (SymbolType)funptr;
}
template <typename T>
inline T toFunPtr(SymbolType ptr) {
  //return static_cast<T> (ptr);
  return (T)ptr;
}
#else
#error
#endif

class LibHandle {
  NO_COPY_CLASS(LibHandle);

 public:
  HandleType handle;
  LibHandle(HandleType handle) : handle(handle) {}
  ~LibHandle() {
    // Note: Currently there is no guarantee that the LibHandle is kept alive until the last OpenCL call has finished. Avoid problems by not unloading the library.
    return;

    if (handle) {
#if OS_UNIX
      dlerror();
      int res = dlclose(handle);
      const char* error = dlerror();
      ASSERT_MSG(res == 0, "dlclose() failed: " + std::string(error));
      ASSERT(error == NULL);
#elif OS_WIN
      Core::WindowsError::check("FreeLibrary", FreeLibrary(handle));
#else
#error
#endif
    }
  }
};

std::shared_ptr<LibHandle> loadOpenCLDriverLibrary() {
  //Core::OStream::getStderr().fprintf("loadOpenCLDriverLibrary()\n");
#if OS_UNIX
  dlerror();
  void* lib = dlopen("libOpenCL.so", RTLD_LAZY);
  const char* error = dlerror();
  if (!lib) throw OpenCL::OpenCLLoadException(error);
  ASSERT(error == NULL);
  return std::make_shared<LibHandle>(lib);
#elif OS_WIN
  HMODULE lib = LoadLibrary("OpenCL");
  if (!lib) Core::WindowsError::error("LoadLibrary (\"OpenCL\")");
  return std::make_shared<LibHandle>(lib);
#else
#error
#endif
}

static std::shared_ptr<LibHandle> getOpenCLDriverLibrary() {
  return Core::staticCache([] { return loadOpenCLDriverLibrary(); });
}

SymbolType loadOpenCLDriverSymbol(const char* name) {
  std::shared_ptr<LibHandle> lib = getOpenCLDriverLibrary();
#if OS_UNIX
  ASSERT(lib->handle != NULL);
  dlerror();
  void* symbol = dlsym(lib->handle, name);
  const char* error = dlerror();
  if (error) throw OpenCL::OpenCLLoadException(error, name);
  return symbol;
#elif OS_WIN
  FARPROC symbol = GetProcAddress(lib->handle, name);
  if (!symbol)
    Core::WindowsError::error("GetProcAddress (OpenCL, \"" + std::string(name) +
                              "\")");
  return symbol;
#else
#error
#endif
}
}  // namespace

// Using STRINGIFY(x) instead of just #x makes sure that macros are expanded
#define STRINGIFY(x) #x

#define OPENCL_LIB_LAZY_DEFINE(n)                                            \
  decltype(::n)* n() {                                                       \
    typedef decltype(::n)* Ptr;                                              \
    return Core::staticCache(                                                \
        [] { return toFunPtr<Ptr>(loadOpenCLDriverSymbol(STRINGIFY(n))); }); \
  }

namespace OpenCL {
OpenCLLoadException::OpenCLLoadException(const std::string& msg) : Exception() {
  this->errorMessage_ = msg;
  this->onResolveSymbol_ = false;
  this->symbol_ = "";
}
OpenCLLoadException::OpenCLLoadException(const std::string& msg,
                                         const std::string& symb)
    : Exception() {
  this->errorMessage_ = msg;
  this->onResolveSymbol_ = true;
  this->symbol_ = symb;
}
OpenCLLoadException::~OpenCLLoadException() {}

std::string OpenCLLoadException::message() const {
  if (!this->onResolveSymbol())
    return "Error loading OpenCL library: " + this->errorMessage();
  else
    return "Error while resolving OpenCL symbol: " + this->symbol() + ": " +
           this->errorMessage();
}

static bool lazyHaveSymbol(const char* name) {
  std::shared_ptr<LibHandle> lib = getOpenCLDriverLibrary();
#if OS_UNIX
  ASSERT(lib->handle != NULL);
  dlerror();
  dlsym(lib->handle, name);  // Ignore result
  const char* error = dlerror();
  return error == NULL;
#elif OS_WIN
  FARPROC symbol = GetProcAddress(lib->handle, name);
  return symbol != NULL;
#else
#error
#endif
}
}  // namespace OpenCL

namespace cl {
namespace OpenCLLibLazy {
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic push
// Ignore deprecation warnings for old OpenCL functions
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

OPENCL_LAZY_SYMBOLS(OPENCL_LIB_LAZY_DEFINE)

#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
}  // namespace OpenCLLibLazy

bool lazyHaveClRetainDevice() {
  return Core::staticCache(
      [] { return OpenCL::lazyHaveSymbol("clRetainDevice"); });
}
}  // namespace cl
