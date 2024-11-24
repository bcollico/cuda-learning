load("@rules_cuda//cuda:defs.bzl", "cuda_library")

package(default_visibility = ["//visibility:public"])

cuda_library(
    name = "kernel3d",
    srcs = ["kernel3d.cu"],
    hdrs = [
        "kernel3d.h",
        "utils.h",
    ],
    copts = ["-dc"],  # allow __device__ function defined in separate compilation unit from kernel.
)

cc_binary(
    name = "main",
    srcs = ["main.cc"],
    deps = [":kernel3d"],
    linkshared = True,
)
