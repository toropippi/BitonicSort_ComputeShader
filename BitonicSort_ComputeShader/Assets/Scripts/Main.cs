using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;


public class Main : MonoBehaviour
{
    struct mystruct
    {
        public float key;//ソートしたい値
        public uint index;//一緒にソートされるindex
    }

    public ComputeShader shader;
    ComputeBuffer gpu_data;
    int kernel_ParallelBitonic_B16;
    int kernel_ParallelBitonic_B8;
    int kernel_ParallelBitonic_B4;
    int kernel_ParallelBitonic_B2;
    int kernel_ParallelBitonic_C4;
    int kernel_ParallelBitonic_C2;
    mystruct[] host_data;
    const int THREADNUM_X = 64;

    //配列の要素数
    int N = 1 << 21;//下限 1<<8,上限 1<<27

    void kernelfindStart()
    {
        kernel_ParallelBitonic_B16 = shader.FindKernel("ParallelBitonic_B16");
        kernel_ParallelBitonic_B8 = shader.FindKernel("ParallelBitonic_B8");
        kernel_ParallelBitonic_B4 = shader.FindKernel("ParallelBitonic_B4");
        kernel_ParallelBitonic_B2 = shader.FindKernel("ParallelBitonic_B2");
        kernel_ParallelBitonic_C4 = shader.FindKernel("ParallelBitonic_C4");
        kernel_ParallelBitonic_C2 = shader.FindKernel("ParallelBitonic_C2");
    }


    void Host_Init()
    {
        for (uint i = 0; i < N; i++)
        {
            host_data[i].key = UnityEngine.Random.Range(-2001.1f, 1.0f);
            host_data[i].index = i;
        }
    }

    void Start()
    {
        host_data = new mystruct[N];//ソートしたいデータ
        gpu_data = new ComputeBuffer(host_data.Length, Marshal.SizeOf(host_data[0]));

        //カーネル初期設定
        kernelfindStart();

        //初期値代入at CPU
        Host_Init();

        // host to device
        gpu_data.SetData(host_data);

        //ソート実装部分
        BitonicSort_fastest(gpu_data, N);
        gpu_data.GetData(host_data);

        //結果表示
        Debug.Log("要素数=" + N);
        resultDebug();
    }

    void BitonicSort_fastest(ComputeBuffer gpu_data,int n)
    {
        //引数をセット
        shader.SetBuffer(kernel_ParallelBitonic_B16, "data", gpu_data);
        shader.SetBuffer(kernel_ParallelBitonic_B8, "data", gpu_data);
        shader.SetBuffer(kernel_ParallelBitonic_B4, "data", gpu_data);
        shader.SetBuffer(kernel_ParallelBitonic_B2, "data", gpu_data);
        shader.SetBuffer(kernel_ParallelBitonic_C4, "data", gpu_data);
        shader.SetBuffer(kernel_ParallelBitonic_C2, "data", gpu_data);

        int nlog = (int)(Mathf.Log(n, 2));
        int B_indx, lninc, inc = 0;
        int kernel_id;

        for (int i = 0; i < nlog; i++)
        {
            lninc = i;
            for (int j = 0; j < i + 1; j++)
            {
                inc = 1 << lninc;
                if (inc <= 128) break;//あとはshared memory内におさまるので

                if (lninc >= 11)
                {
                    B_indx = 16;
                    kernel_id = kernel_ParallelBitonic_B16;
                    lninc = lninc - 4;
                }
                else if (lninc >= 10)
                {
                    B_indx = 8;
                    kernel_id = kernel_ParallelBitonic_B8;
                    lninc = lninc - 3;
                }
                else if (lninc >= 9)
                {
                    B_indx = 4;
                    kernel_id = kernel_ParallelBitonic_B4;
                    lninc = lninc - 2;
                }
                else 
                {
                    B_indx = 2;
                    kernel_id = kernel_ParallelBitonic_B2;
                    lninc = lninc - 1;
                }

                shader.SetInt("inc", inc * 2 / B_indx);
                shader.SetInt("dir", 2 << i);
                shader.Dispatch(kernel_id, N / B_indx / THREADNUM_X, 1, 1);

            }

            //これ以降はshared memoryに収まりそうなサイズなので
            if ((inc == 8) | (inc == 32) | (inc == 128))
            {
                shader.SetInt("inc0", inc);
                shader.SetInt("dir", 2 << i);
                shader.Dispatch(kernel_ParallelBitonic_C4, N / 4 / 64, 1, 1);
            }
            else 
            {
                shader.SetInt("inc0", inc);
                shader.SetInt("dir", 2 << i);
                shader.Dispatch(kernel_ParallelBitonic_C2, N / 2 / 128, 1, 1);
            }
        }//ループの終わり
    }




    void BitonicSort_NoUseSharedMemory(ComputeBuffer gpu_data, int n)
    {
        //引数をセット
        shader.SetBuffer(kernel_ParallelBitonic_B16, "data", gpu_data);
        shader.SetBuffer(kernel_ParallelBitonic_B8, "data", gpu_data);
        shader.SetBuffer(kernel_ParallelBitonic_B4, "data", gpu_data);
        shader.SetBuffer(kernel_ParallelBitonic_B2, "data", gpu_data);

        int nlog = (int)(Mathf.Log(n, 2));
        int B_indx, lninc, inc = 0;
        int kernel_id;

        for (int i = 0; i < nlog; i++)
        {
            lninc = i;
            for (int j = 0; j < i + 1; j++)
            {
                if (lninc < 0) break;
                inc = 1 << lninc;

                if ((lninc >= 3) & (nlog >= 10))
                {
                    B_indx = 16;
                    kernel_id = kernel_ParallelBitonic_B16;
                    lninc = lninc - 4;
                }
                else if ((lninc >= 2) & (nlog >= 9))
                {
                    B_indx = 8;
                    kernel_id = kernel_ParallelBitonic_B8;
                    lninc = lninc - 3;
                }
                else if ((lninc >= 1) & (nlog >= 8))
                {
                    B_indx = 4;
                    kernel_id = kernel_ParallelBitonic_B4;
                    lninc = lninc - 2;
                }
                else
                {
                    B_indx = 2;
                    kernel_id = kernel_ParallelBitonic_B2;
                    lninc = lninc - 1;
                }

                shader.SetInt("inc", inc * 2 / B_indx);
                shader.SetInt("dir", 2 << i);
                shader.Dispatch(kernel_id, N / B_indx / THREADNUM_X, 1, 1);

            }
        }//ループの終わり
    }





    void BitonicSort_normal(ComputeBuffer gpu_data, int n)
    {
        //引数をセット
        shader.SetBuffer(kernel_ParallelBitonic_B2, "data", gpu_data);

        int nlog = (int)(Mathf.Log(n, 2));
        int B_indx, lninc, inc = 0;

        for (int i = 0; i < nlog; i++)
        {
            lninc = i;
            for (int j = 0; j < i + 1; j++)
            {
                inc = 1 << lninc;
                B_indx = 2;
                shader.SetInt("inc", inc * 2 / B_indx);
                shader.SetInt("dir", 2 << i);
                shader.Dispatch(kernel_ParallelBitonic_B2, N / B_indx / THREADNUM_X, 1, 1);
                lninc = lninc - 1;
            }
        }

    }



    void resultDebug()
    {
        // device to host
        gpu_data.GetData(host_data);
        
        Debug.Log("GPU上でソートした結果");
        for (int i = 0; i < Mathf.Min(1024, N); i++)
        {
            Debug.Log("index="+host_data[i].index+" key=" +host_data[i].key);
        }
        
    }


    private void OnDestroy()
    {
        //解放
        gpu_data.Release();
    }
}