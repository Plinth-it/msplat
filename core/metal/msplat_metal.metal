// Metal source manifest. CMake compiles the logical source units below
// separately and links them into default.metallib. Keep this file as an
// orientation point for developers and tests; it is not compiled directly.
#include "msplat_common.metal"
#include "msplat_project.metal"
#include "msplat_raster_backward.metal"
#include "msplat_project_backward.metal"
#include "msplat_project_sh.metal"
#include "msplat_sort.metal"
#include "msplat_loss.metal"
#include "msplat_chunked_raster.metal"
#include "msplat_densify.metal"
