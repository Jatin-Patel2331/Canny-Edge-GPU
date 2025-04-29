/*
 * Copyright (c) 2010-2012 Steffen Kieß
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

#ifndef CORE_OSTREAM_HPP_INCLUDED
#define CORE_OSTREAM_HPP_INCLUDED

#include "OStream.forward.hpp"

// Core::IStream is a reference-counted output stream
// Unlike std::ostream most operations check the stream state and throw an
// exception in case of an error.
//
// Provides a fprintf method which can be used like
// stream.fprintf ("%s %d\n", "foo", 4);
//
// Core::sprintf () works like stream.fprintf but returns a std::string
//
// EPRINTVALS (foo, bar) will print something like "foo = 4, bar = 5" to stderr.

#include <Core/Assert.hpp>
#include <Core/BoostFilesystem.hpp>
#include <Core/Util.hpp>

#include <lib/fmt/include/fmt/format.h>

#include <ostream>
#include <sstream>

#include <boost/format.hpp>

namespace Core {
template <typename... Args>
inline std::string format(Args&&... param) {
  return fmt::format(std::forward<Args>(param)...);
}

class OStream {
  std::shared_ptr<std::ostream> stream;

  struct NoopDeallocator {
    void operator()(UNUSED std::ostream* str) {}
  };

  OStream(std::shared_ptr<std::ostream> stream) : stream(stream) {}

 public:
  OStream(std::ostream* str) : stream(str) { ASSERT(str); }

  bool good() const {
    //ASSERT (stream);
    return stream->good();
  }
  void assertGood() const { ASSERT(good()); }

  std::ostream& operator*() const {
    //ASSERT (stream);
    return *stream;
  }
  std::ostream* operator->() const {
    //ASSERT (stream);
    return stream.get();
  }

  template <typename T>
  const OStream& operator<<(T& t) const {
    assertGood();
    **this << t;
    assertGood();
    return *this;
  }
  template <typename T>
  const OStream& operator<<(const T& t) const {
    assertGood();
    **this << t;
    assertGood();
    return *this;
  }
  // for i/o manipulators
  const OStream& operator<<(std::ostream& (*t)(std::ostream&)) const {
    assertGood();
    **this << t;
    assertGood();
    return *this;
  }

  const OStream& write(const char* data, std::size_t n) const {
    assertGood();
    (*this)->write(data, n);
    assertGood();
    return *this;
  }
  const OStream& write(const std::vector<char>& v) const {
    return write(v.data(), v.size());
  }
  const OStream& write(const std::vector<uint8_t>& v) const {
    return write((const char*)v.data(), v.size());
  }

  template <typename T>
  static void addToFormat(boost::format& format, const T& head) {
    format % head;
  }

  static const boost::format& getValFormat0() {
    static boost::format format("%s = %s, ", std::locale::classic());
    return format;
  }
  static const boost::format& getValFormat() {
    static boost::format format("%s = %s\n", std::locale::classic());
    return format;
  }
  static const boost::format& getValFormatNoNl() {
    static boost::format format("%s = %s", std::locale::classic());
    return format;
  }

 private:
  static std::vector<std::string> splitNames(const std::string& str);

 public:
  void fprintfNoCopy(boost::format& format) const { *this << format; }

  template <typename T>
  void fprintval(const char* name, T value) const {
    fprintf(getValFormat(), name, value);
  }
  template <typename T>
  void fprintvals(UNUSED const boost::format& format0,
                  const boost::format& format,
                  const std::vector<std::string>::const_iterator& firstName,
                  const std::vector<std::string>::const_iterator& lastName,
                  T value) const {
    ASSERT(firstName != lastName);
    ASSERT(firstName + 1 == lastName);
    fprintf(format, *firstName, value);
  }
  template <typename H, typename... T>
  void fprintfNoCopy(boost::format& format, const H& head,
                     const T&... tail) const {
    addToFormat(format, head);
    fprintfNoCopy(format, tail...);
  }

  template <typename... T>
  void fprintf(const boost::format& formatRef, const T&... param) const {
    boost::format format(formatRef);
    fprintfNoCopy(format, param...);
  }
  template <typename... T>
  void fprintf(const char* str, const T&... param) const {
    boost::format format(str, std::locale::classic());
    fprintfNoCopy(format, param...);
  }
  template <typename... T>
  void fprintfL(const char* str, const T&... param) const {
    boost::format format(str);
    fprintfNoCopy(format, param...);
  }
  template <typename... T>
  void fprintf(const char* str, const std::locale& loc,
               const T&... param) const {
    boost::format format(str, loc);
    fprintfNoCopy(format, param...);
  }
  template <typename... T>
  void fprintf(const std::string& str, const T&... param) const {
    boost::format format(str, std::locale::classic());
    fprintfNoCopy(format, param...);
  }
  template <typename... T>
  void fprintfL(const std::string& str, const T&... param) const {
    boost::format format(str);
    fprintfNoCopy(format, param...);
  }
  template <typename... T>
  void fprintf(const std::string& str, const std::locale& loc,
               const T&... param) const {
    boost::format format(str, loc);
    fprintfNoCopy(format, param...);
  }

  template <typename T, typename U0, typename... U>
  void fprintvals(const boost::format& format0, const boost::format& format,
                  const std::vector<std::string>::const_iterator& firstName,
                  const std::vector<std::string>::const_iterator& lastName,
                  T value, U0 values0, const U&... values) const {
    ASSERT(firstName != lastName);
    fprintf(getValFormat0(), *firstName, value);
    fprintvals(format0, format, firstName + 1, lastName, values0, values...);
  }
  template <typename T0, typename... T>
  void fprintvals(const boost::format& format0, const boost::format& format,
                  const std::string& names, T0 values0,
                  const T&... values) const {
    std::vector<std::string> namesV = splitNames(names);
    fprintvals(format0, format, namesV.begin(), namesV.end(), values0,
               values...);
  }

  template <typename... Args>
  inline void print(Args&&... param) {
    *this << Core::format(std::forward<Args>(param)...);
  }
  template <typename... Args>
  inline void println(Args&&... param) {
    *this << Core::format(std::forward<Args>(param)...) << "\n";
  }
  template <typename... Args>
  inline void printlnFlush(Args&&... param) {
    *this << Core::format(std::forward<Args>(param)...) << "\n" << std::flush;
  }

  static OStream get(std::ostream& str) {
    return OStream(std::shared_ptr<std::ostream>(&str, NoopDeallocator()));
  }
  static OStream getStdout();
  static OStream getStderr();

  static OStream tee(const OStream& s1, const OStream& s2);
  static OStream open(const Core::Path& path,
                      std::ios_base::openmode mode = std::ios_base::out);
  static OStream openNull();
#if OS_UNIX
  static OStream openFdDup(int fd);
#endif
};

template <>
inline void OStream::addToFormat<uint8_t>(boost::format& format,
                                          const uint8_t& head) {
  format % (uint16_t)head;
}
template <>
inline void OStream::addToFormat<int8_t>(boost::format& format,
                                         const int8_t& head) {
  format % (int16_t)head;
}

template <typename... T>
inline std::string sprintf(const std::string& str, const T&... param) {
  std::stringstream s;
  OStream::get(s).fprintf(str, param...);
  return s.str();
}
template <typename... T>
inline std::string sprintf(const char* str, const T&... param) {
  std::stringstream s;
  OStream::get(s).fprintf(str, param...);
  return s.str();
}
}  // namespace Core

#define FPRINTF(stream, formatString, ...)                       \
  do {                                                           \
    static boost::format FPRINTF_format("" formatString "",      \
                                        std::locale::classic()); \
    (stream).fprintf(FPRINTF_format, __VA_ARGS__);               \
  } while (0)
#define FPRINTF0(stream, formatString)                           \
  do {                                                           \
    static boost::format FPRINTF_format("" formatString "",      \
                                        std::locale::classic()); \
    (stream).fprintf(FPRINTF_format);                            \
  } while (0)

#define FPRINTFL(stream, formatString, ...)                  \
  do {                                                       \
    static boost::format FPRINTF_format("" formatString ""); \
    (stream).fprintf(FPRINTF_format, __VA_ARGS__);           \
  } while (0)
#define FPRINTFL0(stream, formatString)                      \
  do {                                                       \
    static boost::format FPRINTF_format("" formatString ""); \
    (stream).fprintf(FPRINTF_format);                        \
  } while (0)

#define FPRINTVAL(stream, value) (stream).fprintval(#value, value)
#define EPRINTVAL(value) ::Core::OStream::getStderr().fprintval(#value, value)

#define FPRINTVALS(stream, ...)                                      \
  (stream).fprintvals(::Core::OStream::getValFormat0(),              \
                      ::Core::OStream::getValFormat(), #__VA_ARGS__, \
                      __VA_ARGS__)
#define EPRINTVALS(...)                                                     \
  ::Core::OStream::getStderr().fprintvals(::Core::OStream::getValFormat0(), \
                                          ::Core::OStream::getValFormat(),  \
                                          #__VA_ARGS__, __VA_ARGS__)

#endif  // !CORE_OSTREAM_HPP_INCLUDED
