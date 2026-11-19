# Third-party sources

- The ABH build uses the modified SegMAN-L encoder, decoder and detail-routing code tracked here. The decoder retains its NVIDIA source license header. The build downloads the unchanged Triton scan and seven Oflex source files from the pinned [SegMAN commit](https://github.com/yunxiangfu2001/SegMAN/tree/9ced66ab27f23b0dbc1b0a2fe0ba42e133c55ac8). See the [SegMAN license](https://github.com/yunxiangfu2001/SegMAN/blob/9ced66ab27f23b0dbc1b0a2fe0ba42e133c55ac8/LICENSE).
- MMSegmentation 0.30.0 supplies the base decode head. See the [MMSegmentation license](https://github.com/open-mmlab/mmsegmentation/blob/v0.30.0/LICENSE).
- The three included Oflex files retain the original headers. The selective scan archive attributes the original code to Tri Dao, Albert Gu and the [Mamba project](https://github.com/state-spaces/mamba).
- NATTEN is installed separately from the [v0.17.3 release](https://github.com/SHI-Labs/NATTEN/releases/tag/v0.17.3).
