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

    //配列の要素数
    int N = 1 << 26;//下限 1<<8,上限 1<<27

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
        Debug.Log("要素数=" + N);

        //結果表示
        resultDebug();
    }


    //引数セットとディスパッチ
    void SortDispatch(int kernel_id, int groupnum, int inc = 0, int dir = 0, int inc0 = 0)
    {
        shader.SetInt("inc", inc);
        shader.SetInt("dir", dir);
        shader.SetInt("inc0", inc0);
        shader.Dispatch(kernel_id, groupnum, 1, 1);
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

        for (int i = 0; i < nlog; i++)
        {
            lninc = i;
            for (int j = 0; j < i + 1; j++)
            {
                inc = 1 << lninc;
                if (inc <= 128) break;

                if (lninc >= 11)
                {
                    B_indx = 16;
                    SortDispatch(kernel_ParallelBitonic_B16, N / B_indx / 256, inc * 2 / B_indx, 2 << i);
                    //pbb16(pcl.queue,[int(N / B_indx)],[min(int(N / B_indx), 256)], mem_a, np.uint32(inc * 2 / B_indx), np.uint32(2 << i))
                    lninc = lninc - 4;
                }
                else if (lninc >= 10)
                {
                    B_indx = 8;
                    SortDispatch(kernel_ParallelBitonic_B8, N / B_indx / 256, inc * 2 / B_indx, 2 << i);
                    //pbb8(pcl.queue,[int(N / B_indx)],[min(int(N / B_indx), 256)], mem_a, np.uint32(inc * 2 / B_indx), np.uint32(2 << i))
                    lninc = lninc - 3;
                }
                else if (lninc >= 9)
                {
                    B_indx = 4;
                    SortDispatch(kernel_ParallelBitonic_B4, N / B_indx / 256, inc * 2 / B_indx, 2 << i);
                    //pbb4(pcl.queue,[int(N / B_indx)],[min(int(N / B_indx), 256)], mem_a, np.uint32(inc * 2 / B_indx), np.uint32(2 << i))
                    lninc = lninc - 2;
                }
                else 
                {
                    B_indx = 2;
                    SortDispatch(kernel_ParallelBitonic_B2, N / B_indx / 256, inc * 2 / B_indx, 2 << i);
                    //pbb2(pcl.queue,[int(N/B_indx)],[min(int(N/B_indx),256)],mem_a,np.uint32(inc*2/B_indx),np.uint32(2 << i))
                    lninc = lninc - 1;
                }
                //inc=int(inc/B_indx)
            }

            //これ以降はshared memoryに収まりそうなサイズなので
            if ((inc == 8) | (inc == 32) | (inc == 128))
            {
                SortDispatch(kernel_ParallelBitonic_C4, N / 4 / 64,
                    0,
                    2 << i,
                    inc
                    );
                //pbc4(pcl.queue,[int(N/4)],[min(64,int(N/4))], mem_a, np.uint32(inc), np.uint32(2 << i))
            }
            else 
            {
                SortDispatch(kernel_ParallelBitonic_C2, N / 2 / 128,
                       0,
                       2 << i,
                       inc
                       );
                //pbc2(pcl.queue,[int(N / 2)],[min(128, int(N / 2))], mem_a, np.uint32(inc), np.uint32(2 << i))
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

        for (int i = 0; i < nlog; i++)
        {
            lninc = i;
            for (int j = 0; j < i + 1; j++)
            {
                if (lninc < 0) break;
                inc = 1 << lninc;

                if (lninc >= 3)
                {
                    B_indx = 16;
                    SortDispatch(kernel_ParallelBitonic_B16, N / B_indx / 256, inc * 2 / B_indx, 2 << i);
                    //pbb16(pcl.queue,[int(N / B_indx)],[min(int(N / B_indx), 256)], mem_a, np.uint32(inc * 2 / B_indx), np.uint32(2 << i))
                    lninc = lninc - 4;
                }
                else if (lninc >= 2)
                {
                    B_indx = 8;
                    SortDispatch(kernel_ParallelBitonic_B8, N / B_indx / 256, inc * 2 / B_indx, 2 << i);
                    //pbb8(pcl.queue,[int(N / B_indx)],[min(int(N / B_indx), 256)], mem_a, np.uint32(inc * 2 / B_indx), np.uint32(2 << i))
                    lninc = lninc - 3;
                }
                else if (lninc >= 1)
                {
                    B_indx = 4;
                    SortDispatch(kernel_ParallelBitonic_B4, N / B_indx / 256, inc * 2 / B_indx, 2 << i);
                    //pbb4(pcl.queue,[int(N / B_indx)],[min(int(N / B_indx), 256)], mem_a, np.uint32(inc * 2 / B_indx), np.uint32(2 << i))
                    lninc = lninc - 2;
                }
                else
                {
                    B_indx = 2;
                    SortDispatch(kernel_ParallelBitonic_B2, N / B_indx / 256, inc * 2 / B_indx, 2 << i);
                    //pbb2(pcl.queue,[int(N/B_indx)],[min(int(N/B_indx),256)],mem_a,np.uint32(inc*2/B_indx),np.uint32(2 << i))
                    lninc = lninc - 1;
                }
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
                SortDispatch(kernel_ParallelBitonic_B2, N / B_indx / 256, inc * 2 / B_indx, 2 << i);
                //pbb2(pcl.queue,[int(N/B_indx)],[min(int(N/B_indx),256)],mem_a,np.uint32(inc*2/B_indx),np.uint32(2 << i))
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