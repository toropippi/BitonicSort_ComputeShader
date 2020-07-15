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
    const int THREADNUM_X = 64;//これを変えたらComputeSahder側も変えないといけない

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
        BitonicSort_fastest(gpu_data);
        gpu_data.GetData(host_data);

        //結果表示
        Debug.Log("要素数=" + gpu_data.count);
        resultDebug();
    }

    void BitonicSort_fastest(ComputeBuffer gpu_data)
    {
        int n = gpu_data.count;
        //引数をセット
        shader.SetBuffer(kernel_ParallelBitonic_B16, "data", gpu_data);
        shader.SetBuffer(kernel_ParallelBitonic_B8, "data", gpu_data);
        shader.SetBuffer(kernel_ParallelBitonic_B4, "data", gpu_data);
        shader.SetBuffer(kernel_ParallelBitonic_B2, "data", gpu_data);
        shader.SetBuffer(kernel_ParallelBitonic_C4, "data", gpu_data);
        shader.SetBuffer(kernel_ParallelBitonic_C2, "data", gpu_data);

        int nlog = (int)(Mathf.Log(n, 2));
        int B_indx, inc;
        int kernel_id;

        for (int i = 0; i < nlog; i++)
        {
            inc = 1 << i;
            for (int j = 0; j < i + 1; j++)
            {
                if (inc <= 128) break;//あとはshared memory内におさまるので

                if (inc >= 2048)
                {
                    B_indx = 16;
                    kernel_id = kernel_ParallelBitonic_B16;
                }
                else if (inc >= 1024)
                {
                    B_indx = 8;
                    kernel_id = kernel_ParallelBitonic_B8;
                }
                else if (inc >= 512)
                {
                    B_indx = 4;
                    kernel_id = kernel_ParallelBitonic_B4;
                }
                else 
                {
                    B_indx = 2;
                    kernel_id = kernel_ParallelBitonic_B2;
                }


                shader.SetInt("inc", inc * 2 / B_indx);
                shader.SetInt("dir", 2 << i);
                shader.Dispatch(kernel_id, n / B_indx / THREADNUM_X, 1, 1);
                inc /= B_indx;
            }

            //これ以降はshared memoryに収まりそうなサイズなので
            shader.SetInt("inc0", inc);
            shader.SetInt("dir", 2 << i);
            if ((inc == 8) | (inc == 32) | (inc == 128))
            {
                shader.Dispatch(kernel_ParallelBitonic_C4, n / 4 / 64, 1, 1);
            }
            else 
            {
                shader.Dispatch(kernel_ParallelBitonic_C2, n / 2 / 128, 1, 1);
            }
        }//ループの終わり
    }




    void BitonicSort_NoUseSharedMemory(ComputeBuffer gpu_data)
    {
        int n = gpu_data.count;
        //引数をセット
        shader.SetBuffer(kernel_ParallelBitonic_B16, "data", gpu_data);
        shader.SetBuffer(kernel_ParallelBitonic_B8, "data", gpu_data);
        shader.SetBuffer(kernel_ParallelBitonic_B4, "data", gpu_data);
        shader.SetBuffer(kernel_ParallelBitonic_B2, "data", gpu_data);

        int nlog = (int)(Mathf.Log(n, 2));
        int B_indx, inc;
        int kernel_id;

        for (int i = 0; i < nlog; i++)
        {
            inc = 1 << i;
            for (int j = 0; j < i + 1; j++)
            {
                if (inc == 0) break;

                if ((inc >= 8) & (nlog >= 10))
                {
                    B_indx = 16;
                    kernel_id = kernel_ParallelBitonic_B16;
                }
                else if ((inc >= 4) & (nlog >= 9))
                {
                    B_indx = 8;
                    kernel_id = kernel_ParallelBitonic_B8;
                }
                else if ((inc >= 2) & (nlog >= 8))
                {
                    B_indx = 4;
                    kernel_id = kernel_ParallelBitonic_B4;
                }
                else
                {
                    B_indx = 2;
                    kernel_id = kernel_ParallelBitonic_B2;
                }

                shader.SetInt("inc", inc * 2 / B_indx);
                shader.SetInt("dir", 2 << i);
                shader.Dispatch(kernel_id, n / B_indx / THREADNUM_X, 1, 1);
                inc /= B_indx;
            }
        }//ループの終わり
    }





    void BitonicSort_normal(ComputeBuffer gpu_data)
    {
        int n = gpu_data.count;
        //引数をセット
        shader.SetBuffer(kernel_ParallelBitonic_B2, "data", gpu_data);

        int nlog = (int)(Mathf.Log(n, 2));
        int B_indx, inc;

        for (int i = 0; i < nlog; i++)
        {
            inc = 1 << i;
            for (int j = 0; j < i + 1; j++)
            {
                B_indx = 2;
                shader.SetInt("inc", inc * 2 / B_indx);
                shader.SetInt("dir", 2 << i);
                shader.Dispatch(kernel_ParallelBitonic_B2, n / B_indx / THREADNUM_X, 1, 1);
                inc /= B_indx;
            }
        }
    }







    void resultDebug()
    {
        // device to host
        gpu_data.GetData(host_data);
        
        Debug.Log("GPU上でソートした結果");
        for (int i = 0; i < Mathf.Min(1024, gpu_data.count); i++)
        {
            Debug.Log("index="+host_data[i].index+" key=" +host_data[i].key);
        }

        /*
        int flag = 0;
        for (int i = 1; i < gpu_data.count; i++)
        {
            if (host_data[i].key > host_data[i - 1].key){
                flag = 1;
                break;
            }
        }

        if (flag == 1)
        {
            Debug.Log("ソートできてない！");
        }
        */

    }


    private void OnDestroy()
    {
        //解放
        gpu_data.Release();
    }
}