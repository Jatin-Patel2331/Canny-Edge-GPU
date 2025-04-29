/*
 * Copyright (c) 2013 Steffen Kieß
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

#ifndef CT_DATAFILES_HPP
#define CT_DATAFILES_HPP

#include <Math/Vector3.hpp>
#include <Math/DiagMatrix3.hpp>
#include <Math/Float.hpp>

#include <HDF5/Matlab.hpp>
#include <HDF5/MatlabVector2.hpp>
#include <HDF5/MatlabVector3.hpp>
#include <HDF5/MatlabDiagMatrix3.hpp>
#include <HDF5/DelayedArray.hpp>
#include <HDF5/Array.hpp>

#include <CT/Forward.hpp>
#include <CT/CTFloat.hpp>

#include <boost/scoped_ptr.hpp>

template <typename T>
struct VolumeGen {
  std::string Type; // = "Volume"
  boost::optional<Math::DiagMatrix3<ldouble> > GridSpacing; // distance between two points in X/Y/Z-direction, in m
  boost::optional<Math::Vector3<ldouble> > GridOrigin;
  boost::optional<ldouble> VolumeScalingFactor;
  boost::optional<std::vector<int32_t> > VolumeStorageOrder; // First value: Where is the X dimension stored (1 = normal, -1 = mirrored, 2 = second dim etc.)
  // TODO: make data type dynamic?
  Math::Array<T, 3> Volume;

  Math::ArrayView<const T, 3> transposeVolume (const Math::ArrayView<const T, 3>& view) const {
    if (VolumeStorageOrder)
      return Math::reorderDimensions (view, *VolumeStorageOrder);
    else
      return view;
  }

  Math::ArrayView<const T, 3> transposedUnscaledVolume () const {
    return transposeVolume (Volume.view ());
  }

  void scaleValues (const Math::ArrayView<T, 3>& values) const {
    if (!VolumeScalingFactor)
      return;
    T factor = (T) *VolumeScalingFactor;
    for (size_t z = 0; z < values.template size<2> (); z++)
      for (size_t y = 0; y < values.template size<1> (); y++)
        for (size_t x = 0; x < values.template size<0> (); x++)
          values (x, y, z) *= factor;
  }

private:
  struct NoopDeallocator {
    template <typename U>
    void operator() (UNUSED const U& val) {
    }
  };
public:
  // Rather slow (will transpose volume in memory)
  std::shared_ptr<const Math::Array<T, 3> > transformedTransposedVolume (bool forceCopy = false) const {
    if (forceCopy || VolumeScalingFactor || VolumeStorageOrder) {
      std::shared_ptr<Math::Array<T, 3> > copyPtr = std::make_shared<Math::Array<T, 3> > (transposedUnscaledVolume ());
      scaleValues (copyPtr->view ());
      return copyPtr;
    } else {
      return std::shared_ptr<const Math::Array<T, 3> > (&Volume, NoopDeallocator ());
    }
  }

  std::shared_ptr<const Math::ArrayView<const T, 3> > transformedVolume (bool forceCopy = false) const {
    if (forceCopy || VolumeScalingFactor) {
      //std::shared_ptr<Math::Array<T, 3> > copyPtr = std::make_shared<Math::Array<T, 3> > (Volume);
      std::shared_ptr<std::pair<Math::Array<T, 3>, boost::scoped_ptr<Math::ArrayView<const T, 3> > > > copyPtr = std::make_shared<std::pair<Math::Array<T, 3>, boost::scoped_ptr<Math::ArrayView<const T, 3> > > > ();
      copyPtr->first.recreate (Volume);
      scaleValues (copyPtr->first);
      copyPtr->second.reset (new Math::ArrayView<const T, 3> (transposeVolume (copyPtr->first.view ())));
      //return std::shared_ptr<const Math::ArrayView<const T, 3> > (copyPtr, &copyPtr->view ()); // Only works if Array::view() returns a reference
      return std::shared_ptr<const Math::ArrayView<const T, 3> > (copyPtr, copyPtr->second.get ());
    } else {
      return std::make_shared<const Math::ArrayView<const T, 3> > (transposedUnscaledVolume ());
    }
  }

#define MEMBERS(m)                              \
  m (Type)                                      \
  m (GridSpacing)                               \
  m (GridOrigin)                                \
  m (VolumeScalingFactor)                       \
  m (VolumeStorageOrder)                        \
  m (Volume)
  HDF5_MATLAB_DECLARE_TYPE (VolumeGen, MEMBERS)
#undef MEMBERS
};
typedef VolumeGen<CTFloat> Volume;

template <typename T>
struct VolumeDGen {
  std::string Type; // = "Volume"
  boost::optional<Math::DiagMatrix3<ldouble> > GridSpacing; // distance between two points in X/Y/Z-direction, in m
  boost::optional<Math::Vector3<ldouble> > GridOrigin;
  // TODO: make data type dynamic?
  boost::optional<ldouble> VolumeScalingFactor;
  boost::optional<std::vector<int32_t> > VolumeStorageOrder; // First value: Where is the X dimension stored (1 = normal, -1 = mirrored, 2 = second dim etc.)
  HDF5::DelayedArray<T, 3> Volume;

  std::shared_ptr<VolumeGen<T> > load () const {
    std::shared_ptr<VolumeGen<T> > data = std::make_shared<VolumeGen<T> > ();
    data->Type = Type;
    data->GridSpacing = GridSpacing;
    data->GridOrigin = GridOrigin;
    data->VolumeScalingFactor = VolumeScalingFactor;
    data->VolumeStorageOrder = VolumeStorageOrder;
    data->Volume.recreate (Volume.size);
    Volume.read (data->Volume.view ());
    return data;
  }

  Math::Vector3<size_t> getSize () const {
    Math::Vector3<size_t> s (Volume.size[0], Volume.size[1], Volume.size[2]);
    Math::Vector3<size_t> s2;
    if (!VolumeStorageOrder) {
      s2 = s;
    } else {
      for (size_t i = 0; i < 3; i++) {
        int32_t val = (*VolumeStorageOrder)[i];
        if (val < 0)
          val = -val;
        ASSERT (val > 0 && val <= 3);
        val = val - 1;
        s2[i] = s[val];
      }
    }
    return s2;
  }

#define MEMBERS(m)                              \
  m (Type)                                      \
  m (GridSpacing)                               \
  m (GridOrigin)                                \
  m (VolumeScalingFactor)                       \
  m (VolumeStorageOrder)                        \
  m (Volume)
  HDF5_MATLAB_DECLARE_TYPE (VolumeDGen, MEMBERS)
#undef MEMBERS
};
typedef VolumeDGen<CTFloat> VolumeD;

template <typename T>
struct SimpleConeBeamCTImageSequenceGen {
  std::string Type; // = "SimpleConeBeamCTImageSequence"
  std::string Dimension;
  ldouble DetectorPixelSizeX;
  ldouble DetectorPixelSizeY;
  ldouble DistanceSourceDetector;
  ldouble DistanceSourceAxis;
  std::vector<ldouble> Angle;
  boost::optional<bool> MirroredYAxis;
  // TODO: make data type dynamic?
  Math::Array<T, 3> Image;

private:
  // TODO: dup from Tiff.cpp
  template <typename T2, typename Config, typename Assert>
  static Math::ArrayView<T2, 3, false, Config, Assert> mirrorY (const Math::ArrayView<T2, 3, false, Config, Assert>& view) {
    if (view.template size<1> () == 0)
      return view;
    typename Config::ArithmeticPointer ptr = view.arithData ();
    ptr += (view.template size<1> () - 1) * view.template strideBytes<1> ();
    std::ptrdiff_t stridesBytes[] = { view.template strideBytes<0> (), -view.template strideBytes<1> (), view.template strideBytes<2> () };
    return Math::ArrayView<T2, 3, false, Config, Assert> (Config::template Type<T2>::fromArith (ptr), view.shape (), stridesBytes);
  }
public:

  Math::ArrayView<const T, 3> transformedImage () const {
    if (MirroredYAxis && *MirroredYAxis) {
      return mirrorY (Image.view ());
    } else {
      return Image.view ();
    }
  }

#define MEMBERS(m)                              \
  m (Type)                                      \
  m (Dimension)                                 \
  m (DetectorPixelSizeX)                        \
  m (DetectorPixelSizeY)                        \
  m (DistanceSourceDetector)                    \
  m (DistanceSourceAxis)                        \
  m (Angle)                                     \
  m (MirroredYAxis)                             \
  m (Image)
  HDF5_MATLAB_DECLARE_TYPE (SimpleConeBeamCTImageSequenceGen, MEMBERS)
#undef MEMBERS
};
typedef SimpleConeBeamCTImageSequenceGen<CTFloat> SimpleConeBeamCTImageSequence;

template <typename T>
struct SimpleConeBeamCTImageSequenceDGen {
  std::string Type; // = "SimpleConeBeamCTImageSequence"
  std::string Dimension;
  ldouble DetectorPixelSizeX;
  ldouble DetectorPixelSizeY;
  ldouble DistanceSourceDetector;
  ldouble DistanceSourceAxis;
  std::vector<ldouble> Angle;
  boost::optional<bool> MirroredYAxis;
  // TODO: make data type dynamic?
  HDF5::DelayedArray<T, 3> Image;

#define MEMBERS(m)                              \
  m (Type)                                      \
  m (Dimension)                                 \
  m (DetectorPixelSizeX)                        \
  m (DetectorPixelSizeY)                        \
  m (DistanceSourceDetector)                    \
  m (DistanceSourceAxis)                        \
  m (Angle)                                     \
  m (MirroredYAxis)                             \
  m (Image)
  HDF5_MATLAB_DECLARE_TYPE (SimpleConeBeamCTImageSequenceDGen, MEMBERS)
#undef MEMBERS
};
typedef SimpleConeBeamCTImageSequenceDGen<CTFloat> SimpleConeBeamCTImageSequenceD;

template <typename T>
struct ImageListGen {
  std::string Type; // = "ImageList"
  //Math::DiagMatrix2<ldouble> GridSpacing; // distance between two points in X/Y-direction, in m
  Math::Vector2<ldouble> GridSpacing; // distance between two points in X/Y-direction, in m
  Math::Vector2<ldouble> GridOrigin;
  std::vector<ldouble> ImageId;
  // TODO: make data type dynamic?
  Math::Array<T, 3> Image;

#define MEMBERS(m)                              \
  m (Type)                                      \
  m (GridSpacing)                               \
  m (GridOrigin)                                \
  m (ImageId)                                   \
  m (Image)
  HDF5_MATLAB_DECLARE_TYPE (ImageListGen, MEMBERS)
#undef MEMBERS
};
typedef ImageListGen<CTFloat> ImageList;

template <typename T>
struct ImageListDGen {
  std::string Type; // = "ImageList"
  //Math::DiagMatrix2<ldouble> GridSpacing; // distance between two points in X/Y-direction, in m
  Math::Vector2<ldouble> GridSpacing; // distance between two points in X/Y-direction, in m
  Math::Vector2<ldouble> GridOrigin;
  std::vector<ldouble> ImageId;
  // TODO: make data type dynamic?
  HDF5::DelayedArray<T, 3> Image;

#define MEMBERS(m)                              \
  m (Type)                                      \
  m (GridSpacing)                               \
  m (GridOrigin)                                \
  m (ImageId)                                   \
  m (Image)
  HDF5_MATLAB_DECLARE_TYPE (ImageListDGen, MEMBERS)
#undef MEMBERS
};
typedef ImageListDGen<CTFloat> ImageListD;

#endif // !CT_DATAFILES_HPP
