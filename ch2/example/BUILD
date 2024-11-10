load("@rules_cuda//cuda:defs.bzl", "cuda_library")

package(default_visibility = ["//visibility:public"])

cuda_library(
    name = "example",
    srcs = ["example.cu"],
    hdrs = ["example.h"],
)

cc_binary (
    name = "main",
    srcs = ["main.cc"],
    deps = [":example"],
)
