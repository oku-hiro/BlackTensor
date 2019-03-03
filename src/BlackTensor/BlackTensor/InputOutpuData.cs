using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BlackTensor
{
    /// <summary>
    /// 入力と出力の2次元の値を格納するクラスです。
    /// </summary>
    public class InputOutpuData
    {
        #region プロパティ
        /// <summary>
        /// 入力データを取得します。
        /// </summary>
        public double[][] Input { get; private set; }
        /// <summary>
        /// 出力データを取得します。
        /// </summary>
        public double[][] Output { get; private set; }
        #endregion

        /// <summary>
        /// 指定した値を使用して、初期化します。
        /// </summary>
        /// <param name="dataCount">入力・出力の1次元目の数</param>
        /// <param name="inputCount">入力の2次元目の数</param>
        /// <param name="outputCount">出力の2次元目の数</param>
        public InputOutpuData(int dataCount, int inputCount, int outputCount)
        {
            this.Input = new double[dataCount][];
            this.Output = new double[dataCount][];
            for (var i = 0; i < this.Input.GetLength(0); i++)
            {
                this.Input[i] = new double[inputCount];
            }
            for (var i = 0; i < this.Output.GetLength(0); i++)
            {
                this.Output[i] = new double[outputCount];
            }
        }

        public void SetInputData(double[][] data)
        {
            for (var i = 0; i < this.Input.GetLength(0); i++)
            {
                for (var j = 0; j < this.Input[i].Length; j++)
                {
                    this.Input[i][j] = data[i][j];
                }
            }
        }
        public void SetOutputData(double[][] data)
        {
            for (var i = 0; i < this.Output.GetLength(0); i++)
            {
                for (var j = 0; j < this.Output[i].Length; j++)
                {
                    this.Output[i][j] = data[i][j];
                }
            }
        }
    }
}
