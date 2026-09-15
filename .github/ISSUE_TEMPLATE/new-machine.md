---
name: Add a machine
about: A Mac, a DGX-class box or a GPU the catalog should know
title: "Add machine: <name>"
labels: data
---

Machines live in `machines.json` in the data repository (https://github.com/alexgrankinukr-hash/llm-sizer-data); a pull request there is the fastest path. Either way, the row needs:

- family and chip (with the GPU-core bin when it changes the bandwidth)
- memory options, in the maker's binary gigabytes
- memory bandwidth in GB/s, with the spec page it comes from
- GPU cores per bin
- platform (apple, cuda, rocm) and the interconnect if it links (Thunderbolt 5, ConnectX)
- list prices per memory size and the date, or "no current price"
- sources: the maker's spec page and the store page

Until it is in the catalog, you can type it in under Advanced (memory, bandwidth, platform) and share the link.
