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

#include "OStream.hpp"

#include <Core/BoostFilesystem.hpp>
#include <Core/Error.hpp>
#include <Core/StringUtil.hpp>

#include <fstream>
#include <iostream>

#include <boost/algorithm/string.hpp>
#include <boost/foreach.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/iostreams/tee.hpp>

#if OS_UNIX
#include <boost/iostreams/device/file_descriptor.hpp>
#endif

namespace Core {
namespace {
class TeeStream;
class TeeStreamStuff {
  friend class TeeStream;

  typedef boost::iostreams::tee_device<std::ostream, std::ostream> DeviceType;

 public:
  typedef boost::iostreams::stream<DeviceType> StreamType;

 private:
  OStream str1;
  OStream str2;
  DeviceType device;

  TeeStreamStuff(const OStream& s1, const OStream& s2)
      : str1(s1), str2(s2), device(*str1, *str2) {}
};
class TeeStream : private TeeStreamStuff, public TeeStreamStuff::StreamType {
 public:
  TeeStream(const OStream& s1, const OStream& s2) : TeeStreamStuff(s1, s2) {
    open(device, 0);  // create unbuffered stream
  }
};
}  // namespace

OStream OStream::tee(const OStream& s1, const OStream& s2) {
  return OStream(new TeeStream(s1, s2));
}

OStream OStream::open(const Core::Path& path, std::ios_base::openmode mode) {
  std::string filename = path.BOOST_FILE_STRING;
  errno = 0;
  OStream ret(new std::ofstream(filename.c_str(), mode));
  if (!ret.good()) Core::Error::error("Core::OStream::open(): " + filename);
  return ret;
}

OStream OStream::openNull() {
  return OStream(new boost::iostreams::stream<boost::iostreams::null_sink>(
      (boost::iostreams::null_sink())));
}

#if OS_UNIX
namespace {
class FdStream;
class FdStreamStuff {
  friend class FdStream;

  typedef boost::iostreams::file_descriptor_sink DeviceType;

 public:
  typedef boost::iostreams::stream<DeviceType> StreamType;

 private:
  DeviceType device;

  FdStreamStuff(int fd)
      : device(Core::Error::check("dup", dup(fd)),
               boost::iostreams::file_descriptor_flags::close_handle) {}
};
class FdStream : private FdStreamStuff, public FdStreamStuff::StreamType {
 public:
  FdStream(int fd) : FdStreamStuff(fd) {
    open(device, 0);  // create unbuffered stream
  }
};
}  // namespace

OStream OStream::openFdDup(int fd) { return OStream(new FdStream(fd)); }
#endif

OStream OStream::getStdout() {
  return OStream(std::shared_ptr<std::ostream>(&std::cout, NoopDeallocator()));
}
OStream OStream::getStderr() {
  return OStream(std::shared_ptr<std::ostream>(&std::cerr, NoopDeallocator()));
}

// Split a C expression on commas which are not inside (), [], '' or ""
std::vector<std::string> OStream::splitNames(const std::string& str) {
  std::vector<std::string> res;

  std::string cur;
  int level = 0;
  char quoteChar = 0;
  bool escape = false;
  BOOST_FOREACH (char c, str) {
    if (!quoteChar && level == 0 && c == ',') {
      res.push_back(cur);
      cur = "";
      continue;
    }
    cur += c;
    if (escape) {
      escape = false;
      continue;
    }
    if (quoteChar) {
      if (c == '\\')
        escape = true;
      else if (c == quoteChar)
        quoteChar = 0;
      continue;
    }
    if (c == '(' || c == '[')
      level++;
    else if (level > 0 && (c == ']' || c == ')'))
      level--;
    else if (c == '\'' || c == '"')
      quoteChar = c;
  }
  res.push_back(cur);

  BOOST_FOREACH (std::string& s, res) boost::trim(s);

  return res;
}
}  // namespace Core
