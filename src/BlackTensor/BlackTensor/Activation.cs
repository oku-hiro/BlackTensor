using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BlackTensor
{
    public class Activation : BaseAnalysis
    {
        #region 初期化
        /// <inheritdoc />
        /// <summary>
        /// 初期化します。
        /// </summary>
        public Activation() { }
        /// <inheritdoc />
        /// <summary>
        /// 指定した値を使用して、初期化します。
        /// </summary>
        /// <param name="inputOutputUnit"></param>
        /// <param name="batchSample"></param>
        public Activation(int inputOutputUnit, int batchSample) : base(inputOutputUnit, batchSample) { }
        #endregion


        #region メソッド
        public Tuple<double[][], double[][]> Sigmoid(double[][] flow, double[][] grad)
        {
            this.SetInputGradData(flow, grad);

            for (var b = 0; b < this.BatchSample; b++)
            {
                for (var i = 0; i < this.OutputUnit; i++)
                {
                    this.InputOutputData.Output[b][i] = 1.0 / (1.0 + Math.Exp(-this.InputOutputData.Input[b][i]));
                    this.GradData.Output[b][i] = this.InputOutputData.Output[b][i] * (1.0 - this.InputOutputData.Output[b][i]);
                }
            }

            return new Tuple<double[][], double[][]>(this.InputOutputData.Output, this.GradData.Output);
        }

        public Tuple<double[][], double[][]> ReLU(double[][] flow, double[][] grad)
        {
            this.SetInputGradData(flow, grad);

            for (var b = 0; b < this.BatchSample; b++)
            {
                for (var i = 0; i < this.OutputUnit; i++)
                {
                    if (this.InputOutputData.Input[b][i] < 0.0)
                    {
                        this.InputOutputData.Output[b][i] = 0.0;
                        this.GradData.Output[b][i] = 0.0;
                    }
                    else
                    {
                        this.InputOutputData.Output[b][i] = this.InputOutputData.Input[b][i];
                        this.GradData.Output[b][i] = 1.0;
                    }
                }
            }

            return new Tuple<double[][], double[][]>(this.InputOutputData.Output, this.GradData.Output);
        }

        public Tuple<double[][], double[][]> Tanh(double[][] flow, double[][] grad)
        {
            this.SetInputGradData(flow, grad);

            for (var b = 0; b < this.BatchSample; b++)
            {
                for (var i = 0; i < this.OutputUnit; i++)
                {
                    this.InputOutputData.Output[b][i] = Math.Tanh(this.InputOutputData.Input[b][i]);
                    this.GradData.Output[b][i] = 1.0 - this.InputOutputData.Output[b][i] * this.InputOutputData.Output[b][i];
                }
            }

            return new Tuple<double[][], double[][]>(this.InputOutputData.Output, this.GradData.Output);
        }

        public Tuple<double[][], double[][]> Softmax(double[][] flow, double[][] grad)
        {
            this.SetInputGradData(flow, grad);

            for (var b = 0; b < this.BatchSample; b++)
            {
                for (var i = 0; i < this.OutputUnit; i++)
                {
                    this.InputOutputData.Output[b][i] = Math.Exp(this.InputOutputData.Input[b][i]);
                }

                var sum = this.InputOutputData.Output[b].Sum();

                for (var i = 0; i < this.OutputUnit; i++)
                {
                    this.InputOutputData.Output[b][i] /= sum;
                    this.GradData.Output[b][i] = 1.0;
                }
            }

            return new Tuple<double[][], double[][]>(this.InputOutputData.Output, this.GradData.Output);
        }

        public double[][] DeltaPropagation(double[][] deltaData)
        {
            this.DeltaData.SetInputData(deltaData);

            for (var b = 0; b < this.DeltaData.Output.GetLength(0); b++)
            {
                for (var i = 0; i < this.DeltaData.Output[b].Length; i++)
                {
                    this.DeltaData.Output[b][i] = this.DeltaData.Output[b][i] * this.GradData.Input[b][i];
                }
            }

            return this.DeltaData.Output;
        }
        #endregion
    }
}
