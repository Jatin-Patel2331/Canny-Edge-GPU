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

#ifndef OPENCL_OPENCLLIBLAZY_HPP_INCLUDED
#define OPENCL_OPENCLLIBLAZY_HPP_INCLUDED

#include <Core/Exception.hpp>

#include <OpenCL/cl-patched.hpp>

namespace OpenCL {
/**
 * Exception class for errors while loading the OpenCL library.
 */
class OpenCLLoadException : public Core::Exception {
 private:
  std::string errorMessage_;
  bool onResolveSymbol_;
  std::string symbol_;

 public:
  OpenCLLoadException(const std::string& msg);
  OpenCLLoadException(const std::string& msg, const std::string& symb);
  ~OpenCLLoadException();

  /**
   * @return the error message returned by dlopen() / dlsym()
   */
  const std::string& errorMessage() const { return this->errorMessage_; }

  /**
   * @return false if the error occured while loading the library, true if the
   * error occured while resolving a symbol
   */
  bool onResolveSymbol() const { return this->onResolveSymbol_; }

  /**
   * @return the symbol which caused the error, an empty string if the error
   * occured while loading the library
   */
  const std::string& symbol() const { return this->symbol_; }

  std::string message() const override;
};
}  // namespace OpenCL

#endif  // !OPENCL_OPENCLLIBLAZY_HPP_INCLUDED
