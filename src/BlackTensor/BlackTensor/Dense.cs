using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace BlackTensor
{
    public class Dense : BaseAnalysis
    {
        #region 定数
        private const double Beta1 = 0.9;
        private const double Beta2 = 0.999;
        private const double Gamma = 0.99;
        private readonly double _epsilon = Math.Pow(10.0, -8.0);
        #endregion

        #region プロパティ
        public double Lr { get; }
        #endregion

        private readonly double[][] _w;
        private readonly double[][] _dw;
        private readonly double[][] _m;
        private readonly double[][] _v;

        private double _b1, _b2;

        #region 初期化
        /// <inheritdoc />
        /// <summary>
        /// 初期化します。
        /// </summary>
        public Dense()
        {
            _b1 = Beta1;
            _b2 = Beta2;
        }
        /// <inheritdoc />
        /// <summary>
        /// 指定した値を使用して、初期化します。
        /// </summary>
        /// <param name="inputUnit"></param>
        /// <param name="outputUnit"></param>
        /// <param name="batchSample"></param>
        /// <param name="lr"></param>
        public Dense(int inputUnit, int outputUnit, int batchSample, double lr) : base(inputUnit, outputUnit, batchSample, 1)
        {
            _b1 = Beta1;
            _b2 = Beta2;

            this.Lr = lr;

            var inputUnitPlus = inputUnit + 1;

            for (var i = 0; i < this.InputOutputData.Input.GetLength(0); i++)
            {
                this.InputOutputData.Input[i][0] = 1.0;
            }

            this._m = new double[this.OutputUnit][];
            this._v = new double[this.OutputUnit][];
            this._w = new double[this.OutputUnit][];
            this._dw = new double[this.OutputUnit][];
            for (var i = 0; i < this.OutputUnit; i++)
            {
                this._m[i] = new double[inputUnitPlus];
                this._v[i] = new double[inputUnitPlus];
                this._w[i] = new double[inputUnitPlus];
                this._dw[i] = new double[inputUnitPlus];
            }

            var rnd = new Random();
            for (var j = 0; j < this._w.GetLength(0); j++)
            {
                for (var i = 0; i < this._w[j].Length; i++)
                {
                    _w[j][i] = rnd.NextDouble();
                    _w[j][i] *= Math.Pow(10.0, -5.0);
                }
            }
        }
        #endregion

        #region メソッド
        public Tuple<double[][], double[][]> Process(double[][] flow, double[][] grad)
        {
            this.SetInputGradData(flow, grad);

            for (var b = 0; b < this.InputOutputData.Output.GetLength(0); b++)
            {
                for (var j = 0; j < this.InputOutputData.Output[b].Length; j++)
                {
                    this.InputOutputData.Output[b][j] = 0.0;
                    for (var i = 0; i < this.InputOutputData.Input[b].Length; i++)
                    {
                        this.InputOutputData.Output[b][j] += _w[j][i] * this.InputOutputData.Input[b][i];
                    }
                    this.GradData.Output[b][j] = 1.0;
                }
            }

            return new Tuple<double[][], double[][]>(this.InputOutputData.Output, this.GradData.Output);
        }

        public double[][] DeltaPropagation(double[][] delta)
        {
            this.DeltaData.SetInputData(delta);

            for (var b = 0; b < this.DeltaData.Output.GetLength(0); b++)
            {
                for (var j = 1; j < this.DeltaData.Output[b].Length; j++)
                {
                    this.DeltaData.Output[b][j] = 0.0;
                    
                    for (var i = 0; i < this.DeltaData.Input[b].Length; i++)
                    {
                        this.DeltaData.Output[b][j] += _w[i][j] * this.DeltaData.Input[b][i] * this.GradData.Input[b][j];
                    }
                }
            }

            return this.DeltaData.Output;
        }

        public void BackPropagation()
        {
            _b1 *= Beta1;
            _b2 *= Beta2;
            for (var j = 0; j < this._dw.GetLength(0); j++)
            {
                for (var i = 0; i < this._dw[j].Length; i++)
                {
                    _dw[j][i] = 0.0;
                }
            }

            for (var b = 0; b < this.InputOutputData.Input.GetLength(0); b++)
            {
                for (var j = 0; j < this._dw.GetLength(0); j++)
                {
                    for (var i = 0; i < this._dw[j].Length; i++)
                    {
                        _dw[j][i] += this.DeltaData.Input[b][j] * this.InputOutputData.Input[b][i];
                    }
                }
            }
        }

        public void ADAM()
        {
            for (var j = 0; j < this._w.GetLength(0); j++)
            {
                for (var i = 0; i < this._w[j].Length; i++)
                {
                    _m[j][i] = Beta1 * _m[j][i] + (1.0 - Beta1) * _dw[j][i];
                    _v[j][i] = Beta2 * _v[j][i] + (1.0 - Beta2) * _dw[j][i] * _dw[j][i];
                    var a = _m[j][i] / (1.0 - _b1);
                    var b = _v[j][i] / (1.0 - _b2);
                    _w[j][i] -= Lr * a / (Math.Sqrt(b) + _epsilon);
                }
            }
        }

        public void RmsProp()
        {
            for (var j = 0; j < this._w.GetLength(0); j++)
            {
                for (var i = 0; i < this._w[j].Length; i++)
                {
                    _m[j][i] = Beta1 * _m[j][i] + (1.0 - Gamma) * _dw[j][i] * _dw[j][i];
                    _w[j][i] -= Lr * _dw[j][i] / (Math.Sqrt(_m[j][i]) + _epsilon);
                }
            }
        }

        public void SGD()
        {
            for (var j = 0; j < this._w.GetLength(0); j++)
            {
                for (var i = 0; i < this._w[j].Length; i++)
                {
                    _w[j][i] -= Lr * _dw[j][i];
                }
            }
        }

        public void SaveParameter(int layer)
        {
            using (var sw = new StreamWriter("fc" + (layer + 1)))
            {
                foreach (var w1 in _w)
                {
                    foreach (var w2 in w1)
                    {
                        sw.WriteLine(w2);
                    }
                }
            }
        }
        #endregion

        /// <summary>
        /// 内容を表す文字列を返します。
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            var sb = new StringBuilder();
            sb.AppendLine(nameof(Dense));
            sb.AppendLine($"Input:{this.InputUnit}");
            sb.Append($"Output:{this.OutputUnit}");

            return sb.ToString();
        }
    }
}
