using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace BlackTensor
{
    public class BlackTensor
    {
        private Conv2D[] _cp;
        private Conv2DTranspose[] _ct;
        private Pooling[] _pp;
        private Normalization[] _np;
        private Dense[] _dp;
        private GaussianDistribution[] _gd;
        private Activation[] _ac;

        private int _sqStock = 0;
        private int _cpStock = 0;
        private int _ctStock = 0;
        private int _ppStock = 0;
        private int _npStock = 0;
        private int _dpStock = 0;
        private int _gdStock = 0;
        private int _acStock = 0;

        private int _epochs;
        private int _firstUnit;
        private int _inputUnit;
        private int _outputUnit;
        private int _inputChannel;
        private int _inputX;
        private int _inputY;
        private int _batchSample;
        private int _maxUnit;
        private double _lr;
        private string _stockSequence;
        private string _stockParameter;

        private int[] _batch;
        private int[] _normProcess;
        private int[] _activationProcess;
        private readonly int[] _parameter = new int[4];

        private double[][] _flow;
        private double[][] _teacher;
        private double[][] _grad;
        private double[][] _delta;
        private double[] _totalError;
        private double[] _output;
        private string[] _sequence;

        public BlackTensor() { }

        #region Layer
        private void Layer_Conv2d(int n)
        {
            _cp = new Conv2D[n];
            for (var i = 0; i < n; i++)
            {
                _cp[i] = new Conv2D();
            }
        }

        private void Layer_Conv2d_Transpose(int n)
        {
            _ct = new Conv2DTranspose[n];
            for (var i = 0; i < n; i++)
            {
                _ct[i] = new Conv2DTranspose();
            }
        }

        private void Layer_Pooling(int n)
        {
            _pp = new Pooling[n];
            for (var i = 0; i < n; i++)
            {
                _pp[i] = new Pooling();
            }
        }

        private void Layer_Dense(int n)
        {
            _dp = new Dense[n];
            for (var i = 0; i < n; i++)
            {
                _dp[i] = new Dense();
            }
        }

        private void Layer_Norm(int n)
        {
            _np = new Normalization[n];
            for (var i = 0; i < n; i++)
            {
                _np[i] = new Normalization();
            }
        }

        private void Layer_Gaussian_Distribution(int n)
        {
            _gd = new GaussianDistribution[n];
            for (var i = 0; i < n; i++)
            {
                _gd[i] = new GaussianDistribution();
            }
        }

        private void Layer_Activation(int n)
        {
            _ac = new Activation[n];
            for (var i = 0; i < n; i++)
            {
                _ac[i] = new Activation();
            }
        }
        #endregion

        public void Conv2d(int stride, int filterChannel, int filterX, int filterY)
        {
            _cpStock += 1;
            _sqStock += 1;
            _stockSequence += "conv2d,";
            _stockParameter += stride + ",";
            _stockParameter += filterChannel + ",";
            _stockParameter += filterX + ",";
            _stockParameter += filterY + ",";
        }

        public void Conv2dTranspose(int filterChannel, int filterX, int filterY)
        {
            _ctStock += 1;
            _sqStock += 1;
            _stockSequence += "conv2d_t,";
            _stockParameter += filterChannel + ",";
            _stockParameter += filterX + ",";
            _stockParameter += filterY + ",";
        }

        public void Pooling(int sizeX, int sizeY)
        {
            _ppStock += 1;
            _sqStock += 1;
            _stockSequence += "pooling,";
            _stockParameter += sizeX + ",";
            _stockParameter += sizeY + ",";
        }

        public void Dense(int unit, int channel, int outputX, int outputY)
        {
            _dpStock += 1;
            _sqStock += 1;
            _stockSequence += "dense,";
            _stockParameter += unit + ",";
            _stockParameter += channel + ",";
            _stockParameter += outputX + ",";
            _stockParameter += outputY + ",";
        }

        public void Normalization(int sizeX, int sizeY, int process)
        {
            _npStock += 1;
            _sqStock += 1;
            _stockSequence += "norm,";
            _stockParameter += sizeX + ",";
            _stockParameter += sizeY + ",";
            _stockParameter += process + ",";
        }

        public void GaussianDistribution()
        {
            _gdStock += 1;
            _sqStock += 1;
            _stockSequence += "gaudis,";
        }

        public void Activation(int function)
        {
            _acStock += 1;
            _sqStock += 1;
            _stockSequence += "activation,";
            _stockParameter += function.ToString() + ",";
        }

        public void Setting()
        {
            _batch = new int[_batchSample];
            _totalError = new double[_epochs];
            _sequence = new string[_sqStock];

            ExtractSequence();

            if (_cpStock > 0)
            {
                Layer_Conv2d(_cpStock);
            }

            if (_ctStock > 0)
            {
                Layer_Conv2d_Transpose(_ctStock);
            }

            if (_ppStock > 0)
            {
                Layer_Pooling(_ppStock);
            }

            if (_npStock > 0)
            {
                _normProcess = new int[_npStock];
                Layer_Norm(_npStock);
            }

            if (_dpStock > 0)
            {
                Layer_Dense(_dpStock);
            }

            if (_gdStock > 0)
            {
                Layer_Gaussian_Distribution(_gdStock);
            }

            if (_acStock > 0)
            {
                _activationProcess = new int[_acStock];
                Layer_Activation(_acStock);
            }

            var cpStep = 0;
            var ctStep = 0;
            var ppStep = 0;
            var npStep = 0;
            var dpStep = 0;
            var gdStep = 0;
            var acStep = 0;
            var position = 0;

            for (var i = 0; i < _sqStock; i++)
            {
                switch (_sequence[i])
                {
                    case "conv2d":
                        position = ExtractParameter(position, 4);
                        Conv2d_Setting(cpStep);
                        cpStep++;
                        break;
                    case "conv2d_t":
                        position = ExtractParameter(position, 3);
                        Conv2d_Transpose_Setting(ctStep);
                        ctStep++;
                        break;
                    case "pooling":
                        position = ExtractParameter(position, 2);
                        Pooling_Setting(ppStep);
                        ppStep++;
                        break;
                    case "norm":
                        position = ExtractParameter(position, 3);
                        Norm_Setting(npStep);
                        npStep++;
                        break;
                    case "dense":
                        position = ExtractParameter(position, 4);
                        Dense_Setting(dpStep);
                        dpStep++;
                        break;
                    case "gaudis":
                        Gaussian_Distribution_Setting(gdStep);
                        gdStep++;
                        break;
                    case "activation":
                        position = ExtractParameter(position, 1);
                        Activation_Setting(acStep);
                        acStep++;
                        break;
                }
            }
            _outputUnit = _inputUnit;

            _flow = new double[_batchSample][];
            _grad = new double[_batchSample][];
            _delta = new double[_batchSample][];
            _teacher = new double[_batchSample][];

            for (var i = 0; i < _batchSample; i++)
            {
                _flow[i] = new double[_maxUnit];
                _grad[i] = new double[_maxUnit];
                _delta[i] = new double[_maxUnit];
                _teacher[i] = new double[_outputUnit];
            }
            _output = new double[_outputUnit];
        }

        public void Learning(LearningParameter lp)
        {
            _lr = lp.lr;
            _inputChannel = lp.input_channel;
            _inputX = lp.input_x;
            _inputY = lp.input_y;
            _inputUnit = lp.dense_unit + lp.input_channel * lp.input_x * lp.input_y;
            _batchSample = lp.batch_sample;
            _epochs = lp.epochs;
            _firstUnit = _inputUnit;

            Setting();
            Summary();

            long totalTime = 0;

            var group = 0;
            for (var k = 0; k < lp.epochs; k++)
            {
                var sw = new Stopwatch();
                sw.Start();

                for (var i = 0; i < _batchSample; i++)
                {
                    _batch[i] = i + _batchSample * group;
                }

                for (var b = 0; b < _batchSample; b++)
                {
                    for (var i = 0; i < _outputUnit; i++)
                    {
                        _teacher[b][i] = lp.output_data[_batch[b]][i];
                    }

                    for (var i = 0; i < _firstUnit; i++)
                    {
                        _flow[b][i] = lp.input_data[_batch[b]][i];
                    }
                }

                Network();

                _totalError[k] = 0.0;
                for (var b = 0; b < _batchSample; b++)
                {
                    for (var i = 0; i < _outputUnit; i++)
                    {
                        _totalError[k] += (_flow[b][i] - _teacher[b][i]) * (_flow[b][i] - _teacher[b][i]);
                    }
                }

                BackPropagation();

                switch (lp.optimizer)
                {
                    case 0:
                        SGD();
                        break;
                    case 1:
                        ADAM();
                        break;
                    case 2:
                        RmsProp();
                        break;
                }

                group++;
                if (group == lp.data_sample / _batchSample)
                {
                    group = 0;
                }

                sw.Stop();

                totalTime += sw.ElapsedMilliseconds;
                var avgTime = totalTime / (k + 1);
                var completeTime = avgTime * (lp.epochs - k - 1);

                Console.WriteLine($"{k + 1}/{lp.epochs}：Error = {_totalError[k]}\tTime = {sw.ElapsedMilliseconds}[ms]\tComplete = {completeTime / 1000}[s]");

                //Console.WriteLine(_totalError[k]);
            }

            Console.WriteLine($"TotalTime：{totalTime}[ms]");

            SaveParameter();

            using (var sw = new StreamWriter("total_error"))
            {
                for (var i = 0; i < lp.epochs; i++)
                {
                    sw.WriteLine(_totalError[i]);
                }
            }
        }

        public double[] Evaluate(double[] inputData)
        {
            _batchSample = 1;
            for (var i = 0; i < _firstUnit; i++)
            {
                _flow[0][i] = inputData[i];
            }

            Network();

            for (var i = 0; i < _outputUnit; i++)
            {
                _output[i] = _flow[0][i];
            }

            return _output;
        }

        private void ADAM()
        {
            for (var i = 0; i < _cpStock; i++)
            {
                _cp[i].ADAM();
            }

            for (var i = 0; i < _ctStock; i++)
            {
                _ct[i].ADAM();
            }

            for (var i = 0; i < _dpStock; i++)
            {
                _dp[i].ADAM();
            }
        }

        private void RmsProp()
        {
            for (var i = 0; i < _cpStock; i++)
            {
                _cp[i].RmsProp();
            }

            for (var i = 0; i < _ctStock; i++)
            {
                _ct[i].RmsProp();
            }

            for (var i = 0; i < _dpStock; i++)
            {
                _dp[i].RmsProp();
            }
        }

        private void SGD()
        {
            for (var i = 0; i < _cpStock; i++)
            {
                _cp[i].SGD();
            }

            for (var i = 0; i < _ctStock; i++)
            {
                _ct[i].SGD();
            }

            for (var i = 0; i < _dpStock; i++)
            {
                _dp[i].SGD();
            }
        }

        #region Network

        private void Network()
        {
            var cpStep = 0;
            var ctStep = 0;
            var ppStep = 0;
            var npStep = 0;
            var dpStep = 0;
            var gdStep = 0;
            var acStep = 0;

            for (var i = 0; i < _sqStock; i++)
            {
                Tuple<double[][], double[][]> result;

                switch (_sequence[i])
                {
                    case "conv2d":
                        result = _cp[cpStep].Process(_flow, _grad);
                        break;
                    case "conv2d_t":
                        result = _ct[ctStep].Process(_flow, _grad);
                        ctStep++;
                        break;
                    case "pooling":
                        result = _pp[ppStep].Process(_flow, _grad);
                        ppStep++;
                        break;
                    case "norm":
                        result = this.NormNetwork(npStep);
                        npStep++;
                        break;
                    case "dense":
                        result = _dp[dpStep].Process(_flow, _grad);
                        dpStep++;
                        break;
                    case "gaudis":
                        result = _gd[gdStep].Process(_flow, _grad);
                        gdStep++;
                        break;
                    case "activation":
                        result = this.ActivationNetwork(acStep);
                        acStep++;
                        break;
                    default:
                        continue;
                }

                if (result == null) continue;

                for (var b = 0; b < result.Item1.GetLength(0); b++)
                {
                    for (var j = 0; j < result.Item1[b].Length; j++)
                    {
                        _flow[b][j] = result.Item1[b][j];
                        _grad[b][j] = result.Item2[b][j];
                    }
                }
            }
        }

        private Tuple<double[][], double[][]> NormNetwork(int layer)
        {
            Tuple<double[][], double[][]> result;

            switch (_normProcess[layer])
            {
                case 0:
                    result = _np[layer].Subtractive(_flow, _grad);
                    break;
                case 1:
                    result = _np[layer].Divisive(_flow, _grad);
                    break;
                case 2:
                    result = _np[layer].Batch1D(_flow, _grad);
                    break;
                default:
                    return null;
            }

            return result;
        }

        private Tuple<double[][], double[][]> ActivationNetwork(int layer)
        {
            _ac[layer].SetInputGradData(_flow, _grad);

            Tuple<double[][], double[][]> result;

            switch (_activationProcess[layer])
            {
                case 0:
                    result = _ac[layer].Sigmoid(_flow, _grad);
                    break;
                case 1:
                    result = _ac[layer].ReLU(_flow, _grad);
                    break;
                case 2:
                    result = _ac[layer].Softmax(_flow, _grad);
                    break;
                case 3:
                    result = _ac[layer].Tanh(_flow, _grad);
                    break;
                default:
                    return null;
            }

            return result;
        }

        #endregion

        #region Extract
        private void ExtractSequence()
        {
            var start = 0;
            for (var j = 0; j < _sqStock; j++)
            {
                var strData = "";
                for (var i = start; i < _stockSequence.Length; i++)
                {
                    if (_stockSequence.Substring(i, 1) != ",")
                    {
                        strData += _stockSequence.Substring(i, 1);
                    }
                    else
                    {
                        start = i + 1;
                        break;
                    }
                }
                _sequence[j] = strData;
            }
        }

        private int ExtractParameter(int position, int count)
        {
            for (var j = 0; j < count; j++)
            {
                var strData = "";
                for (var i = position; i < _stockParameter.Length; i++)
                {
                    if (_stockParameter.Substring(i, 1) != ",")
                    {
                        strData += _stockParameter.Substring(i, 1);
                    }
                    else
                    {
                        position = i + 1;
                        break;
                    }
                }
                _parameter[j] = int.Parse(strData);
            }
            return position;
        }
        #endregion

        #region BackPropagation
        private void BackPropagation()
        {
            var cpStep = _cpStock - 1;
            var ctStep = _ctStock - 1;
            var ppStep = _ppStock - 1;
            var npStep = _npStock - 1;
            var dpStep = _dpStock - 1;
            var gdStep = _gdStock - 1;
            var acStep = _acStock - 1;

            for (var b = 0; b < _batchSample; b++)
            {
                for (var i = 0; i < _outputUnit; i++)
                {
                    _delta[b][i] = (_flow[b][i] - _teacher[b][i]) * _grad[b][i]; //二乗誤差
                }
            }

            for (var i = _sqStock - 1; i >= 0; i--)
            {
                double[][] result;
                var offset = 0;

                switch (_sequence[i])
                {
                    case "conv2d":
                        result = _cp[cpStep].DeltaPropagation(_delta);
                        cpStep--;
                        break;
                    case "conv2d_t":
                        result = _ct[ctStep].DeltaPropagation(_delta);
                        ctStep--;
                        break;
                    case "pooling":
                        result = _pp[ppStep].DeltaPropagation(_delta);
                        ppStep--;
                        break;
                    case "norm":
                        result = _np[npStep].DeltaPropagation(_delta);
                        npStep--;
                        break;
                    case "dense":
                        result = _dp[dpStep].DeltaPropagation(_delta);
                        offset = 1;
                        dpStep--;
                        break;
                    case "gaudis":
                        result = _gd[gdStep].DeltaPropagation(_delta);
                        gdStep--;
                        break;
                    case "activation":
                        result = _ac[acStep].DeltaPropagation(_delta);
                        acStep--;
                        break;
                    default:
                        continue;
                }

                for (var j = 0; j < result.GetLength(0); j++)
                {
                    for (var k = 0; k < result[j].Length - offset; k++)
                    {
                        this._delta[j][k] = result[j][k + offset];
                    }
                }
            }

            for (var i = 0; i < _cpStock; i++)
            {
                _cp[i].BackPropagation();
            }

            for (var i = 0; i < _ctStock; i++)
            {
                _ct[i].BackPropagation();
            }

            for (var i = 0; i < _dpStock; i++)
            {
                _dp[i].BackPropagation();
            }
        }
        #endregion

        #region Setting
        private void Conv2d_Setting(int layer)
        {
            _cp[layer] = new Conv2D(_batchSample, _inputX, _inputY, _parameter[2], _parameter[3], _parameter[0], _inputChannel, _parameter[1], _lr);

            _inputX = _cp[layer].Output2D.X;
            _inputY = _cp[layer].Output2D.Y;
            _inputChannel = _cp[layer].OutputChannel;
            _inputUnit = _cp[layer].OutputUnit;
            MaxUnit(_cp[layer].InputUnit, _cp[layer].OutputUnit);
        }

        private void Conv2d_Transpose_Setting(int layer)
        {
            _ct[layer] = new Conv2DTranspose(_batchSample, _inputX, _inputY, _parameter[1], _parameter[2], _inputChannel, _parameter[0], _lr);

            _inputX = _ct[layer].Output2D.X;
            _inputY = _ct[layer].Output2D.Y;
            _inputChannel = _ct[layer].OutputChannel;
            _inputUnit = _ct[layer].OutputUnit;
            MaxUnit(_ct[layer].InputUnit, _ct[layer].OutputUnit);
        }

        private void Pooling_Setting(int layer)
        {
            _pp[layer] = new Pooling(_batchSample, _inputX, _inputY, _parameter[0], _parameter[1], _inputChannel);
            _inputChannel = _pp[layer].OutputChannel;
            _inputX = _pp[layer].Output2D.X;
            _inputY = _pp[layer].Output2D.Y;
            _inputUnit = _pp[layer].OutputUnit;
            MaxUnit(_pp[layer].InputUnit, _pp[layer].OutputUnit);
        }

        private void Norm_Setting(int layer)
        {
            _np[layer] = new Normalization(_inputUnit, _batchSample, _inputX, _inputY, _parameter[0], _parameter[1], _inputChannel);

            _normProcess[layer] = _parameter[2];
            _inputUnit = _np[layer].OutputUnit;
            MaxUnit(_np[layer].InputUnit, _np[layer].OutputUnit);
        }

        private void Dense_Setting(int layer)
        {
            _dp[layer] = new Dense(_inputUnit, _parameter[0], _batchSample, _lr);

            _inputChannel = _parameter[1];
            _inputX = _parameter[2];
            _inputY = _parameter[3];
            _inputUnit = _dp[layer].OutputUnit;
            MaxUnit(_dp[layer].InputUnit, _dp[layer].OutputUnit);
        }

        private void Gaussian_Distribution_Setting(int layer)
        {
            _gd[layer] = new GaussianDistribution(_inputUnit, _batchSample);

            _inputChannel = 0;
            _inputX = 0;
            _inputY = 0;
            _inputUnit = _gd[layer].OutputUnit;
            MaxUnit(_gd[layer].InputUnit, _gd[layer].OutputUnit);
        }

        private void Activation_Setting(int layer)
        {
            _activationProcess[layer] = _parameter[0];
            _ac[layer] = new Activation(_inputUnit, _batchSample);
            MaxUnit(_ac[layer].InputUnit, _ac[layer].OutputUnit);
        }
        #endregion

        private void SaveParameter()
        {
            for (var i = 0; i < _cpStock; i++)
            {
                _cp[i].SaveParameter(i);
            }

            for (var i = 0; i < _dpStock; i++)
            {
                _dp[i].SaveParameter(i);
            }

            for (var i = 0; i < _ctStock; i++)
            {
                _ct[i].SaveParameter(i);
            }
        }

        private void MaxUnit(int unit1, int unit2)
        {
            var max = Math.Max(unit1, unit2);
            this._maxUnit = Math.Max(this._maxUnit, max);
        }

        private void Summary()
        {
            var cpStep = 0;
            var ctStep = 0;
            var ppStep = 0;
            var npStep = 0;
            var dpStep = 0;
            var gdStep = 0;

            for (var i = 0; i < _sqStock; i++)
            {
                switch (_sequence[i])
                {
                    case "conv2d":
                        Console.WriteLine(_cp[cpStep]);
                        cpStep++;
                        break;
                    case "conv2d_t":
                        Console.WriteLine(_ct[ctStep]);
                        ctStep++;
                        break;
                    case "pooling":
                        Console.WriteLine(_pp[ppStep]);
                        ppStep++;
                        break;
                    case "norm":
                        Console.WriteLine(_np[npStep]);
                        npStep++;
                        break;
                    case "dense":
                        Console.WriteLine(_dp[dpStep]);
                        dpStep++;
                        break;
                    case "gaudis":
                        Console.WriteLine(_gd[gdStep]);
                        gdStep++;
                        break;
                    default:
                        continue;
                }

                Console.WriteLine("******************************************************************************************************");
            }
        }
    }
}