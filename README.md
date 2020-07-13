## Introduction

バイトニックソートのコンピュートシェーダー移植版
indexとkeyを一緒にソートします


Bitonic Sort implemented in ComputeShader.
Sort index and key together.



## 高速化

高速化は
http://www.bealto.com/gpu-sorting_parallel-bitonic-1.html
を参考にしました。


### Only B2

1threadあたり2箇所Load,Storeを行います。

![gpuimpliment1](https://user-images.githubusercontent.com/44022497/87314023-a8a16000-c55d-11ea-9353-6dd51890e7d6.png)

一番標準的な実装です。実行時間(ms)の表です。

|要素数/カーネル名|B2|
|---|---|
|65536|1|
|131072|1|
|262144|2|
|524288|4|
|1048576|10|
|2097152|19|
|4194304|39|
|8388608|83|
|16777216|179|
|33554432|382|
|67108864|818|
|134217728|1744|


### B2C2

Shared Memoryを使った高速化版です。

![shared1](https://user-images.githubusercontent.com/44022497/87314088-b8b93f80-c55d-11ea-8e55-df8c4850bfc5.png)

青の部分はグローバルメモリへ書き込みをしないでShared memoryに書き込み使いまわしている部分です。
グローバルメモリのアクセスを抑えたことにより高速化できます。

|要素数/カーネル名|B2|B2C2|
|---|---|---|
|65536|1|0|
|131072|1|1|
|262144|2|1|
|524288|4|3|
|1048576|10|8|
|2097152|19|11|
|4194304|39|24|
|8388608|83|51|
|16777216|179|109|
|33554432|382|236|
|67108864|818|508|
|134217728|1744|1095|


### B2B4B8B16C2C4

1threadあたり4箇所Load,Storeを行うことでグローバルメモリのアクセス回数を減らします。
![gpuimpliment2](https://user-images.githubusercontent.com/44022497/87314155-c66ec500-c55d-11ea-9bd9-a8227274e079.png)
これを8,16と増やすことでさらなる高速化ができました。
Shared memory内も1threadあたり4箇所Load,Storeを行うことでShared memoryへのアクセス回数を減らします。

|要素数/カーネル名|B2|B2C2|B2B4B8B16C2C4|
|---|---|---|---|
|65536|1|0|0|
|131072|1|1|0|
|262144|2|1|1|
|524288|4|3|2|
|1048576|10|8|5|
|2097152|19|11|7|
|4194304|39|24|14|
|8388608|83|51|29|
|16777216|179|109|60|
|33554432|382|236|127|
|67108864|818|508|264|
|134217728|1744|1095|552|


最終進化形だとここまで速くなります。最初と比べ3倍以上高速化できました！！わーい