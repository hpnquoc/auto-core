## ColorMaps in OpenCV

[Enumerations](#enumerations) | [Functions](#functions)

ColorMaps in OpenCV

### Detailed Description
--------------------

The human perception isn't built for observing fine changes in grayscale images. Human eyes are more sensitive to observing changes between colors, so you often need to recolor your grayscale images to get a clue about them. OpenCV now comes with various colormaps to enhance the visualization in your computer vision application.

In OpenCV you only need apply ColorMap to apply a colormap on a given image. The following sample code reads the path to an image from command line, applies a Jet colormap on it and shows the result:

``` C
#include <opencv2/core.hpp>

#include <opencv2/imgproc.hpp>

#include <opencv2/imgcodecs.hpp>

#include <opencv2/highgui.hpp>

using namespace cv;

#include <iostream>

using namespace std;

int main(int argc, const char *argv[])
{
 // We need an input image. (can be grayscale or color)
 if (argc < 2)
 {
 cerr << "We need an image to process here. Please run: colorMap [path_to_image]" << endl;
 return -1;
 }
 Mat img_in = imread(argv[1]);
 if(img_in.empty())
 {
 cerr << "Sample image (" << argv[1] << ") is empty. Please adjust your path, so it points to a valid input image!" << endl;
 return -1;
 }
 // Holds the colormap version of the image:
 Mat img_color;
 // Apply the colormap:
 applyColorMap(img_in, img_color, COLORMAP_JET);
 // Show the result:
 imshow("colorMap", img_color);
 waitKey(0);
 return 0;
}
```

### Enumerations
------------

enum  
```cpp
enum ColormapTypes {
    COLORMAP_AUTUMN           = 0,
    COLORMAP_BONE             = 1,
    COLORMAP_JET              = 2,
    COLORMAP_WINTER           = 3,
    COLORMAP_RAINBOW          = 4,
    COLORMAP_OCEAN            = 5,
    COLORMAP_SUMMER           = 6,
    COLORMAP_SPRING           = 7,
    COLORMAP_COOL             = 8,
    COLORMAP_HSV              = 9,
    COLORMAP_PINK             = 10,
    COLORMAP_HOT              = 11,
    COLORMAP_PARULA           = 12,
    COLORMAP_MAGMA            = 13,
    COLORMAP_INFERNO          = 14,
    COLORMAP_PLASMA           = 15,
    COLORMAP_VIRIDIS          = 16,
    COLORMAP_CIVIDIS          = 17,
    COLORMAP_TWILIGHT         = 18,
    COLORMAP_TWILIGHT_SHIFTED = 19,
    COLORMAP_TURBO            = 20,
    COLORMAP_DEEPGREEN        = 21
};
```


### Enumerator

| Colormap Type      | Python Equivalent            | Example Image                                       |
|--------------------|------------------------------|-----------------------------------------------------|
| AUTUMN             | cv.COLORMAP_AUTUMN           | ![](../data/opencv/colorscale_autumn.jpg)           |
| BONE               | cv.COLORMAP_BONE             | ![](../data/opencv/colorscale_bone.jpg)             |
| JET                | cv.COLORMAP_JET              | ![](../data/opencv/colorscale_jet.jpg)              |
| WINTER             | cv.COLORMAP_WINTER           | ![](../data/opencv/colorscale_winter.jpg)           |
| RAINBOW            | cv.COLORMAP_RAINBOW          | ![](../data/opencv/colorscale_rainbow.jpg)          |
| OCEAN              | cv.COLORMAP_OCEAN            | ![](../data/opencv/colorscale_ocean.jpg)            |
| SUMMER             | cv.COLORMAP_SUMMER           | ![](../data/opencv/colorscale_summer.jpg)           |
| SPRING             | cv.COLORMAP_SPRING           | ![](../data/opencv/colorscale_spring.jpg)           |
| COOL               | cv.COLORMAP_COOL             | ![](../data/opencv/colorscale_cool.jpg)             |
| HSV                | cv.COLORMAP_HSV              | ![](../data/opencv/colorscale_hsv.jpg)              |
| PINK               | cv.COLORMAP_PINK             | ![](../data/opencv/colorscale_pink.jpg)             |
| HOT                | cv.COLORMAP_HOT              | ![](../data/opencv/colorscale_hot.jpg)              |
| PARULA             | cv.COLORMAP_PARULA           | ![](../data/opencv/colorscale_parula.jpg)           |
| MAGMA              | cv.COLORMAP_MAGMA            | ![](../data/opencv/colorscale_magma.jpg)            |
| INFERNO            | cv.COLORMAP_INFERNO          | ![](../data/opencv/colorscale_inferno.jpg)          |
| PLASMA             | cv.COLORMAP_PLASMA           | ![](../data/opencv/colorscale_plasma.jpg)           |
| VIRIDIS            | cv.COLORMAP_VIRIDIS          | ![](../data/opencv/colorscale_viridis.jpg)          |
| CIVIDIS            | cv.COLORMAP_CIVIDIS          | ![](../data/opencv/colorscale_cividis.jpg)          |
| TWILIGHT           | cv.COLORMAP_TWILIGHT         | ![](../data/opencv/colorscale_twilight.jpg)         |
| TWILIGHT_SHIFTED   | cv.COLORMAP_TWILIGHT_SHIFTED | ![](../data/opencv/colorscale_twilight_shifted.jpg) |
| TURBO              | cv.COLORMAP_TURBO            | ![](../data/opencv/colorscale_turbo.jpg)            |
| DEEPGREEN          | cv.COLORMAP_DEEPGREEN        | ![](../data/opencv/colorscale_deepgreen.jpg)        |

### Functions
#### imread

```cpp
Mat cv::imread(const String &filename, int flags = IMREAD_COLOR)
```

Reads an image from a file.

**Parameters:**
- `filename`: Name of the file to be loaded.
- `flags`: Flags specifying the color type of a loaded image.

**Example:**

```cpp
Mat img = imread("path_to_image", IMREAD_COLOR);
```

#### imshow

```cpp
void cv::imshow(const String &winname, InputArray mat)
```

Displays an image in the specified window.

**Parameters:**
- `winname`: Name of the window.
- `mat`: Image to be shown.

**Example:**

```cpp
imshow("Display window", img);
waitKey(0);
```

#### waitKey

```cpp
int cv::waitKey(int delay = 0)
```

Waits for a pressed key.

**Parameters:**
- `delay`: Delay in milliseconds. 0 is the special value that means "forever".

**Example:**

```cpp
waitKey(0);
```

#### applyColorMap

```cpp
void cv::applyColorMap(InputArray src, OutputArray dst, int colormap)
```

Applies a colormap on a given image.

**Parameters:**
- `src`: The source image, grayscale or color.
- `dst`: The destination image.
- `colormap`: The colormap to apply, see `cv::ColormapTypes`.

**Example:**

```cpp
Mat img_in = imread("path_to_image");
Mat img_color;
applyColorMap(img_in, img_color, COLORMAP_JET);
imshow("colorMap", img_color);
waitKey(0);
```