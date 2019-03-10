using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace BlackTensor
{
    public class GaussianDistribution : BaseAnalysis
    {
        #region 定数
        private readonly Random _rnd = new Random();
        private const double Pi = Math.PI;
        #endregion 

        #region 初期化
        /// <inheritdoc />
        /// <summary>
        /// 初期化します。
        /// </summary>
        public GaussianDistribution() {}
        /// <inheritdoc />
        /// <summary>
        /// 指定した値を使用して、初期化します。
        /// </summary>
        /// <param name="inputUnit"></param>
        /// <param name="batchSample"></param>
        public GaussianDistribution(int inputUnit, int batchSample) : base(inputUnit, inputUnit/2, batchSample) { }
        #endregion

        #region メソッド
        public Tuple<double[][], double[][]> Process(double[][] flow, double[][] grad)
        {
            this.SetInputGradData(flow, grad);

            var sw = new Stopwatch();
            sw.Start();

            for (var b = 0; b < this.BatchSample; b++)
            {
                var z = Math.Sqrt(-2.0 * Math.Log(_rnd.NextDouble())) * Math.Cos(2.0 * Pi * _rnd.NextDouble());
                for (var i = 0; i < this.InputUnit; i++)
                {
                    this.InputOutputData.Output[b][i] = this.InputOutputData.Input[b][2 * i] + z * this.InputOutputData.Input[b][2 * i + 1];
                    this.GradData.Output[b][i] = 1.0;
                }
            }

            sw.Stop();
            Debug.WriteLine($"{nameof(GaussianDistribution)}.{nameof(this.Process)}：{sw.ElapsedMilliseconds}[ms]");

            return new Tuple<double[][], double[][]>(this.InputOutputData.Output, this.GradData.Output);
        }

        public double[][] DeltaPropagation(double[][] delta)
        {
            this.DeltaData.SetInputData(delta);

            var sw = new Stopwatch();
            sw.Start();

            for (var b = 0; b < this.DeltaData.Output.GetLength(0); b++)
            {
                for (var j = 0; j < this.DeltaData.Output[b].Length; j++)
                {
                    for (var i = 2 * j; i < 2 * (j + 1); i++)
                    {
                        this.DeltaData.Output[b][i] = this.DeltaData.Input[b][j] * this.GradData.Input[b][i];
                    }
                }
            }

            sw.Stop();
            Debug.WriteLine($"{nameof(DeltaPropagation)}.{nameof(this.DeltaPropagation)}：{sw.ElapsedMilliseconds}[ms]");

            return this.DeltaData.Output;
        }
        #endregion

        /// <summary>
        /// 内容を表す文字列を返します。
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            var sb = new StringBuilder();
            sb.AppendLine(nameof(GaussianDistribution));
            sb.AppendLine($"Input:{this.InputUnit}");
            sb.Append($"Output:{this.OutputUnit}");

            return sb.ToString();
        }
    }
}