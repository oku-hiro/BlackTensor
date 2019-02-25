using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace BlackTensor
{
    public class BlackTensor
    {
        Conv2d[] cp;
        Conv2dTranspose[] ct;
        Pooling[] pp;
        Normalization[] np;
        Dense[] dp;
        GaussianDistribution[] gd;
        Activation[] ac;

        int sq_stock = 0;
        int cp_stock = 0;
        int ct_stock = 0;
        int pp_stock = 0;
        int np_stock = 0;
        int dp_stock = 0;
        int gd_stock = 0;
        int ac_stock = 0;

        int epochs;
        int first_unit;
        int input_unit;
        int output_unit;
        int input_channel;
        int input_x;
        int input_y;
        int batch_sample;
        int max_unit;
        double lr;
        string stock_sequence;
        string stock_parameter;

        int[] batch;
        int[] norm_process;
        int[] activation_process;
        int[] parameter = new int[4];

        public double[][] flow;
        double[][] teacher;
        double[][] grad;
        double[][] delta;
        double[] total_error;
        double[] output;
        string[] sequence;

        public BlackTensor()
        { }

        private void Layer_Conv2d(int n)
        {
            cp = new Conv2d[n];
            for (int i = 0; i < n; i++)
            {
                cp[i] = new Conv2d();
            }
        }

        private void Layer_Conv2d_Transpose(int n)
        {
            ct = new Conv2dTranspose[n];
            for (int i = 0; i < n; i++)
            {
                ct[i] = new Conv2dTranspose();
            }
        }

        private void Layer_Pooling(int n)
        {
            pp = new Pooling[n];
            for (int i = 0; i < n; i++)
            {
                pp[i] = new Pooling();
            }
        }

        private void Layer_Dense(int n)
        {
            dp = new Dense[n];
            for (int i = 0; i < n; i++)
            {
                dp[i] = new Dense();
            }
        }

        private void Layer_Norm(int n)
        {
            np = new Normalization[n];
            for (int i = 0; i < n; i++)
            {
                np[i] = new Normalization();
            }
        }

        private void Layer_Gaussian_Distribution(int n)
        {
            gd = new GaussianDistribution[n];
            for (int i = 0; i < n; i++)
            {
                gd[i] = new GaussianDistribution();
            }
        }

        private void Layer_Activation(int n)
        {
            ac = new Activation[n];
            for (int i = 0; i < n; i++)
            {
                ac[i] = new Activation();
            }
        }

        public void Conv2d(int stride, int filter_channel, int filter_x, int filter_y)
        {
            cp_stock += 1;
            sq_stock += 1;
            stock_sequence += "conv2d,";
            stock_parameter += stride.ToString() + ",";
            stock_parameter += filter_channel.ToString() + ",";
            stock_parameter += filter_x.ToString() + ",";
            stock_parameter += filter_y.ToString() + ",";
        }

        public void Conv2dTranspose(int filter_channel, int filter_x, int filter_y)
        {
            ct_stock += 1;
            sq_stock += 1;
            stock_sequence += "conv2d_t,";
            stock_parameter += filter_channel.ToString() + ",";
            stock_parameter += filter_x.ToString() + ",";
            stock_parameter += filter_y.ToString() + ",";
        }

        public void Pooling(int size_x, int size_y)
        {
            pp_stock += 1;
            sq_stock += 1;
            stock_sequence += "pooling,";
            stock_parameter += size_x.ToString() + ",";
            stock_parameter += size_y.ToString() + ",";
        }

        public void Dense(int unit, int channel, int output_x, int output_y)
        {
            dp_stock += 1;
            sq_stock += 1;
            stock_sequence += "dense,";
            stock_parameter += unit.ToString() + ",";
            stock_parameter += channel.ToString() + ",";
            stock_parameter += output_x.ToString() + ",";
            stock_parameter += output_y.ToString() + ",";
        }

        public void Normalization(int size_x, int size_y, int process)
        {
            np_stock += 1;
            sq_stock += 1;
            stock_sequence += "norm,";
            stock_parameter += size_x.ToString() + ",";
            stock_parameter += size_y.ToString() + ",";
            stock_parameter += process.ToString() + ",";
        }

        public void GaussianDistribution()
        {
            gd_stock += 1;
            sq_stock += 1;
            stock_sequence += "gaudis,";
        }

        public void Activation(int function)
        {
            ac_stock += 1;
            sq_stock += 1;
            stock_sequence += "activation,";
            stock_parameter += function.ToString() + ",";
        }

        public void Setting()
        {
            batch = new int[batch_sample];
            total_error = new double[epochs];
            sequence = new string[sq_stock];

            Extract_Sequence();

            if (cp_stock > 0)
            {
                Layer_Conv2d(cp_stock);
            }

            if (ct_stock > 0)
            {
                Layer_Conv2d_Transpose(ct_stock);
            }

            if (pp_stock > 0)
            {
                Layer_Pooling(pp_stock);
            }

            if (np_stock > 0)
            {
                norm_process = new int[np_stock];
                Layer_Norm(np_stock);
            }

            if (dp_stock > 0)
            {
                Layer_Dense(dp_stock);
            }

            if (gd_stock > 0)
            {
                Layer_Gaussian_Distribution(gd_stock);
            }

            if (ac_stock > 0)
            {
                activation_process = new int[ac_stock];
                Layer_Activation(ac_stock);
            }

            int cp_step = 0;
            int ct_step = 0;
            int pp_step = 0;
            int np_step = 0;
            int dp_step = 0;
            int gd_step = 0;
            int ac_step = 0;
            int position = 0;

            for (int i = 0; i < sq_stock; i++)
            {
                if (sequence[i] == "conv2d")
                {
                    position = Extract_Parameter(position, 4);
                    Conv2d_Setting(cp_step);
                    cp_step += 1;
                }
                if (sequence[i] == "conv2d_t")
                {
                    position = Extract_Parameter(position, 3);
                    Conv2d_Transpose_Setting(ct_step);
                    ct_step += 1;
                }
                else if (sequence[i] == "pooling")
                {
                    position = Extract_Parameter(position, 2);
                    Pooling_Setting(pp_step);
                    pp_step += 1;
                }
                else if (sequence[i] == "norm")
                {
                    position = Extract_Parameter(position, 3);
                    Norm_Setting(np_step);
                    np_step += 1;
                }
                else if (sequence[i] == "dense")
                {
                    position = Extract_Parameter(position, 4);
                    Dense_Setting(dp_step);
                    dp_step += 1;
                }
                else if (sequence[i] == "gaudis")
                {
                    Gaussian_Distribution_Setting(gd_step);
                    gd_step += 1;
                }
                else if (sequence[i] == "activation")
                {
                    position = Extract_Parameter(position, 1);
                    Activation_Setting(ac_step);
                    ac_step += 1;
                }
            }
            output_unit = input_unit;

            flow = new double[batch_sample][];
            grad = new double[batch_sample][];
            delta = new double[batch_sample][];
            teacher = new double[batch_sample][];

            for (int i = 0; i < batch_sample; i++)
            {
                flow[i] = new double[max_unit];
                grad[i] = new double[max_unit];
                delta[i] = new double[max_unit];
                teacher[i] = new double[output_unit];
            }
            output = new double[output_unit];
        }

        public void Learning(LearningParameter lp)
        {
            lr = lp.lr;
            input_channel = lp.input_channel;
            input_x = lp.input_x;
            input_y = lp.input_y;
            input_unit = lp.dense_unit + lp.input_channel * lp.input_x * lp.input_y;
            batch_sample = lp.batch_sample;
            epochs = lp.epochs;
            first_unit = input_unit;

            Setting();
            Summary();

            int group = 0;
            for (int k = 0; k < lp.epochs; k++)
            {
                for (int i = 0; i < batch_sample; i++)
                {
                    batch[i] = i + batch_sample * group;
                }

                for (int b = 0; b < batch_sample; b++)
                {
                    for (int i = 0; i < output_unit; i++)
                    {
                        teacher[b][i] = lp.output_data[batch[b]][i];
                    }

                    for (int i = 0; i < first_unit; i++)
                    {
                        flow[b][i] = lp.input_data[batch[b]][i];
                    }
                }

                Network();

                total_error[k] = 0.0;
                for (int b = 0; b < batch_sample; b++)
                {
                    for (int i = 0; i < output_unit; i++)
                    {
                        total_error[k] += (flow[b][i] - teacher[b][i]) * (flow[b][i] - teacher[b][i]);
                    }
                }

                Back_Propagation();

                if (lp.optimizer == 0)
                {
                    SGD();
                }
                else if (lp.optimizer == 1)
                {
                    ADAM();
                }
                else if (lp.optimizer == 2)
                {
                    RmsProp();
                }

                group += 1;
                if (group == lp.data_sample / batch_sample)
                {
                    group = 0;
                }

                Console.WriteLine(total_error[k]);
            }

            SaveParameter();

            StreamWriter sw = new StreamWriter("total_error");
            for (int i = 0; i < lp.epochs; i++)
            {
                sw.WriteLine(total_error[i]);
            }
            sw.Close();
        }

        public double[] Evaluate(double[] input_data)
        {
            batch_sample = 1;
            for (int i = 0; i < first_unit; i++)
            {
                flow[0][i] = input_data[i];
            }

            Network();

            for (int i = 0; i < output_unit; i++)
            {
                output[i] = flow[0][i];
            }

            return output;
        }

        private void ADAM()
        {
            for (int i = 0; i < cp_stock; i++)
            {
                cp[i].ADAM();
            }

            for (int i = 0; i < ct_stock; i++)
            {
                ct[i].ADAM();
            }

            for (int i = 0; i < dp_stock; i++)
            {
                dp[i].ADAM();
            }
        }

        private void RmsProp()
        {
            for (int i = 0; i < cp_stock; i++)
            {
                cp[i].RmsProp();
            }

            for (int i = 0; i < ct_stock; i++)
            {
                ct[i].RmsProp();
            }

            for (int i = 0; i < dp_stock; i++)
            {
                dp[i].RmsProp();
            }
        }

        private void SGD()
        {
            for (int i = 0; i < cp_stock; i++)
            {
                cp[i].SGD();
            }

            for (int i = 0; i < ct_stock; i++)
            {
                ct[i].SGD();
            }

            for (int i = 0; i < dp_stock; i++)
            {
                dp[i].SGD();
            }
        }

        private void Network()
        {
            int cp_step = 0;
            int ct_step = 0;
            int pp_step = 0;
            int np_step = 0;
            int dp_step = 0;
            int gd_step = 0;
            int ac_step = 0;

            for (int i = 0; i < sq_stock; i++)
            {
                if (sequence[i] == "conv2d")
                {
                    Conv2d_Network(cp_step);
                    cp_step += 1;
                }
                else if (sequence[i] == "conv2d_t")
                {
                    Conv2d_Transpose_Network(ct_step);
                    ct_step += 1;
                }
                else if (sequence[i] == "pooling")
                {
                    Pooling_Network(pp_step);
                    pp_step += 1;
                }
                else if (sequence[i] == "norm")
                {
                    Norm_Network(np_step);
                    np_step += 1;
                }
                else if (sequence[i] == "dense")
                {
                    Dense_Network(dp_step);
                    dp_step += 1;
                }
                else if (sequence[i] == "gaudis")
                {
                    Gaussian_Distribution_Network(gd_step);
                    gd_step += 1;
                }
                else if (sequence[i] == "activation")
                {
                    Activation_Network(ac_step);
                    ac_step += 1;
                }
            }
        }

        private void Conv2d_Network(int layer)
        {
            for (int b = 0; b < batch_sample; b++)
            {
                for (int i = 0; i < cp[layer].input_unit; i++)
                {
                    cp[layer].input[b][i] = flow[b][i];
                    cp[layer].pre_grad[b][i] = grad[b][i];
                }
            }
            cp[layer].Process();
            for (int b = 0; b < batch_sample; b++)
            {
                for (int i = 0; i < cp[layer].output_unit; i++)
                {
                    flow[b][i] = cp[layer].output[b][i];
                    grad[b][i] = cp[layer].this_grad[b][i];
                }
            }
        }

        private void Conv2d_Transpose_Network(int layer)
        {
            for (int b = 0; b < batch_sample; b++)
            {
                for (int i = 0; i < ct[layer].input_unit; i++)
                {
                    ct[layer].input[b][i] = flow[b][i];
                    ct[layer].pre_grad[b][i] = grad[b][i];
                }
            }
            ct[layer].Process();
            for (int b = 0; b < batch_sample; b++)
            {
                for (int i = 0; i < ct[layer].output_unit; i++)
                {
                    flow[b][i] = ct[layer].output[b][i];
                    grad[b][i] = ct[layer].this_grad[b][i];
                }
            }
        }

        private void Pooling_Network(int layer)
        {
            for (int b = 0; b < batch_sample; b++)
            {
                for (int i = 0; i < pp[layer].input_unit; i++)
                {
                    pp[layer].input[b][i] = flow[b][i];
                    pp[layer].pre_grad[b][i] = grad[b][i];
                }
            }
            pp[layer].Process();
            for (int b = 0; b < batch_sample; b++)
            {
                for (int i = 0; i < pp[layer].output_unit; i++)
                {
                    flow[b][i] = pp[layer].output[b][i];
                    grad[b][i] = pp[layer].this_grad[b][i];
                }
            }
        }

        private void Norm_Network(int layer)
        {
            for (int b = 0; b < batch_sample; b++)
            {
                for (int i = 0; i < np[layer].input_unit; i++)
                {
                    np[layer].input[b][i] = flow[b][i];
                    np[layer].pre_grad[b][i] = grad[b][i];
                }
            }

            if (norm_process[layer] == 0)
            {
                np[layer].Subtractive();
            }
            else if (norm_process[layer] == 1)
            {
                np[layer].Divisive();
            }
            else if (norm_process[layer] == 2)
            {
                np[layer].Batch1D();
            }

            for (int b = 0; b < batch_sample; b++)
            {
                for (int i = 0; i < np[layer].output_unit; i++)
                {
                    flow[b][i] = np[layer].output[b][i];
                    grad[b][i] = np[layer].this_grad[b][i];
                }
            }
        }

        private void Dense_Network(int layer)
        {
            for (int b = 0; b < batch_sample; b++)
            {
                for (int i = 0; i < dp[layer].input_unit; i++)
                {
                    dp[layer].input[b][i + 1] = flow[b][i];
                    dp[layer].pre_grad[b][i + 1] = grad[b][i];
                }
            }
            dp[layer].Process();
            for (int b = 0; b < batch_sample; b++)
            {
                for (int i = 0; i < dp[layer].output_unit; i++)
                {
                    flow[b][i] = dp[layer].output[b][i];
                    grad[b][i] = dp[layer].this_grad[b][i];
                }
            }
        }

        private void Gaussian_Distribution_Network(int layer)
        {
            for (int b = 0; b < batch_sample; b++)
            {
                for (int i = 0; i < gd[layer].input_unit; i++)
                {
                    gd[layer].input[b][i] = flow[b][i];
                    gd[layer].pre_grad[b][i] = grad[b][i];
                }
            }
            gd[layer].Process();
            for (int b = 0; b < batch_sample; b++)
            {
                for (int i = 0; i < gd[layer].output_unit; i++)
                {
                    flow[b][i] = gd[layer].output[b][i];
                    grad[b][i] = gd[layer].this_grad[b][i];
                }
            }
        }

        private void Activation_Network(int layer)
        {
            for (int b = 0; b < batch_sample; b++)
            {
                for (int i = 0; i < ac[layer].input_unit; i++)
                {
                    ac[layer].input[b][i] = flow[b][i];
                    ac[layer].pre_grad[b][i] = grad[b][i];
                }
            }

            if (activation_process[layer] == 0)
            {
                ac[layer].Sigmoid();
            }
            else if (activation_process[layer] == 1)
            {
                ac[layer].ReLU();
            }
            else if (activation_process[layer] == 2)
            {
                ac[layer].Softmax();
            }
            else if (activation_process[layer] == 3)
            {
                ac[layer].Tanh();
            }

            for (int b = 0; b < batch_sample; b++)
            {
                for (int i = 0; i < ac[layer].output_unit; i++)
                {
                    flow[b][i] = ac[layer].output[b][i];
                    grad[b][i] = ac[layer].this_grad[b][i];
                }
            }
        }

        private void Back_Propagation()
        {
            int cp_step = cp_stock - 1;
            int ct_step = ct_stock - 1;
            int pp_step = pp_stock - 1;
            int np_step = np_stock - 1;
            int dp_step = dp_stock - 1;
            int gd_step = gd_stock - 1;
            int ac_step = ac_stock - 1;

            for (int b = 0; b < batch_sample; b++)
            {
                for (int i = 0; i < output_unit; i++)
                {
                    delta[b][i] = (flow[b][i] - teacher[b][i]) * grad[b][i]; //二乗誤差
                }
            }

            for (int i = sq_stock - 1; i >= 0; i--)
            {
                if (sequence[i] == "conv2d")
                {
                    Conv2d_Delta_Propagation(cp_step);
                    cp_step -= 1;
                }
                else if (sequence[i] == "conv2d_t")
                {
                    Conv2d_Transpose_Delta_Propagation(ct_step);
                    ct_step -= 1;
                }
                else if (sequence[i] == "pooling")
                {
                    Pooling_Delta_Propagation(pp_step);
                    pp_step -= 1;
                }
                else if (sequence[i] == "norm")
                {
                    Norm_Delta_Propagation(np_step);
                    np_step -= 1;
                }
                else if (sequence[i] == "dense")
                {
                    Dense_Delta_Propagation(dp_step);
                    dp_step -= 1;
                }
                else if (sequence[i] == "gaudis")
                {
                    Gaussian_Distribution_Delta_Propagation(gd_step);
                    gd_step -= 1;
                }
                else if (sequence[i] == "activation")
                {
                    Activation_Delta_Propagation(ac_step);
                    ac_step -= 1;
                }
            }

            for (int i = 0; i < cp_stock; i++)
            {
                cp[i].BackPropagation();
            }

            for (int i = 0; i < ct_stock; i++)
            {
                ct[i].BackPropagation();
            }

            for (int i = 0; i < dp_stock; i++)
            {
                dp[i].Back_Propagation();
            }
        }

        private void Extract_Sequence()
        {
            int start = 0;
            for (int j = 0; j < sq_stock; j++)
            {
                string str_data = "";
                for (int i = start; i < stock_sequence.Length; i++)
                {
                    if (stock_sequence.Substring(i, 1) != ",")
                    {
                        str_data += stock_sequence.Substring(i, 1);
                    }
                    else
                    {
                        start = i + 1;
                        break;
                    }
                }
                sequence[j] = str_data;
            }
        }

        private int Extract_Parameter(int position, int count)
        {
            for (int j = 0; j < count; j++)
            {
                string str_data = "";
                for (int i = position; i < stock_parameter.Length; i++)
                {
                    if (stock_parameter.Substring(i, 1) != ",")
                    {
                        str_data += stock_parameter.Substring(i, 1);
                    }
                    else
                    {
                        position = i + 1;
                        break;
                    }
                }
                parameter[j] = Int32.Parse(str_data);
            }
            return position;
        }

        private void Conv2d_Delta_Propagation(int layer)
        {
            for (int b = 0; b < batch_sample; b++)
            {
                for (int i = 0; i < cp[layer].output_unit; i++)
                {
                    cp[layer].this_delta[b][i] = delta[b][i];
                }
            }
            cp[layer].DeltaPropagation();
            for (int b = 0; b < batch_sample; b++)
            {
                for (int i = 0; i < cp[layer].input_unit; i++)
                {
                    delta[b][i] = cp[layer].next_delta[b][i];
                }
            }
        }

        private void Conv2d_Transpose_Delta_Propagation(int layer)
        {
            for (int b = 0; b < batch_sample; b++)
            {
                for (int i = 0; i < ct[layer].output_unit; i++)
                {
                    ct[layer].this_delta[b][i] = delta[b][i];
                }
            }
            ct[layer].DeltaPropagation();
            for (int b = 0; b < batch_sample; b++)
            {
                for (int i = 0; i < ct[layer].input_unit; i++)
                {
                    delta[b][i] = ct[layer].next_delta[b][i];
                }
            }
        }

        private void Pooling_Delta_Propagation(int layer)
        {
            for (int b = 0; b < batch_sample; b++)
            {
                for (int i = 0; i < pp[layer].output_unit; i++)
                {
                    pp[layer].this_delta[b][i] = delta[b][i];
                }
            }
            pp[layer].Delta_Propagation();
            for (int b = 0; b < batch_sample; b++)
            {
                for (int i = 0; i < pp[layer].input_unit; i++)
                {
                    delta[b][i] = pp[layer].next_delta[b][i];
                }
            }
        }

        private void Norm_Delta_Propagation(int layer)
        {
            for (int b = 0; b < batch_sample; b++)
            {
                for (int i = 0; i < np[layer].output_unit; i++)
                {
                    np[layer].this_delta[b][i] = delta[b][i];
                }
            }
            np[layer].Delta_Propagation();
            for (int b = 0; b < batch_sample; b++)
            {
                for (int i = 0; i < np[layer].input_unit; i++)
                {
                    delta[b][i] = np[layer].next_delta[b][i];
                }
            }
        }

        private void Dense_Delta_Propagation(int layer)
        {
            for (int b = 0; b < batch_sample; b++)
            {
                for (int i = 0; i < dp[layer].output_unit; i++)
                {
                    dp[layer].this_delta[b][i] = delta[b][i];
                }
            }
            dp[layer].Delta_Propagation();
            for (int b = 0; b < batch_sample; b++)
            {
                for (int i = 0; i < dp[layer].input_unit; i++)
                {
                    delta[b][i] = dp[layer].next_delta[b][i + 1];
                }
            }
        }

        private void Gaussian_Distribution_Delta_Propagation(int layer)
        {
            for (int b = 0; b < batch_sample; b++)
            {
                for (int i = 0; i < gd[layer].output_unit; i++)
                {
                    gd[layer].this_delta[b][i] = delta[b][i];
                }
            }
            gd[layer].Delta_Propagation();
            for (int b = 0; b < batch_sample; b++)
            {
                for (int i = 0; i < gd[layer].input_unit; i++)
                {
                    delta[b][i] = gd[layer].next_delta[b][i];
                }
            }
        }

        private void Activation_Delta_Propagation(int layer)
        {
            for (int b = 0; b < batch_sample; b++)
            {
                for (int i = 0; i < ac[layer].output_unit; i++)
                {
                    ac[layer].this_delta[b][i] = delta[b][i];
                }
            }
            ac[layer].Delta_Propagation();
            for (int b = 0; b < batch_sample; b++)
            {
                for (int i = 0; i < ac[layer].input_unit; i++)
                {
                    delta[b][i] = ac[layer].next_delta[b][i];
                }
            }
        }

        private void Conv2d_Setting(int layer)
        {
            cp[layer].lr = lr;
            cp[layer].stride = parameter[0];
            cp[layer].filter_channel = parameter[1];
            cp[layer].filter_x = parameter[2];
            cp[layer].filter_y = parameter[3];
            cp[layer].input_channel = input_channel;
            cp[layer].input_x = input_x;
            cp[layer].input_y = input_y;
            cp[layer].batch_sample = batch_sample;
            cp[layer].Setting();
            input_x = cp[layer].output_x;
            input_y = cp[layer].output_y;
            input_channel = cp[layer].output_channel;
            input_unit = cp[layer].output_unit;
            MaxUnit(cp[layer].input_unit, cp[layer].output_unit);
        }

        private void Conv2d_Transpose_Setting(int layer)
        {
            ct[layer].lr = lr;
            ct[layer].filter_channel = parameter[0];
            ct[layer].filter_x = parameter[1];
            ct[layer].filter_y = parameter[2];
            ct[layer].input_channel = input_channel;
            ct[layer].input_x = input_x;
            ct[layer].input_y = input_y;
            ct[layer].batch_sample = batch_sample;
            ct[layer].Setting();
            input_x = ct[layer].output_x;
            input_y = ct[layer].output_y;
            input_channel = ct[layer].output_channel;
            input_unit = ct[layer].output_unit;
            MaxUnit(ct[layer].input_unit, ct[layer].output_unit);
        }

        private void Pooling_Setting(int layer)
        {
            pp[layer].size_x = parameter[0];
            pp[layer].size_y = parameter[1];
            pp[layer].input_channel = input_channel;
            pp[layer].input_x = input_x;
            pp[layer].input_y = input_y;
            pp[layer].batch_sample = batch_sample;
            pp[layer].Setting();
            input_channel = pp[layer].output_channel;
            input_x = pp[layer].output_x;
            input_y = pp[layer].output_y;
            input_unit = pp[layer].output_unit;
            MaxUnit(pp[layer].input_unit, pp[layer].output_unit);
        }

        private void Norm_Setting(int layer)
        {
            np[layer].size_x = parameter[0];
            np[layer].size_y = parameter[1];
            norm_process[layer] = parameter[2];
            np[layer].input_channel = input_channel;
            np[layer].input_x = input_x;
            np[layer].input_y = input_y;
            np[layer].input_unit = input_unit;
            np[layer].batch_sample = batch_sample;
            np[layer].Setting();
            input_unit = np[layer].output_unit;
            MaxUnit(np[layer].input_unit, np[layer].output_unit);
        }

        private void Dense_Setting(int layer)
        {
            dp[layer].lr = lr;
            dp[layer].output_unit = parameter[0];
            dp[layer].input_unit = input_unit;
            dp[layer].batch_sample = batch_sample;
            dp[layer].Setting();
            input_channel = parameter[1];
            input_x = parameter[2];
            input_y = parameter[3];
            input_unit = dp[layer].output_unit;
            MaxUnit(dp[layer].input_unit, dp[layer].output_unit);
        }

        private void Gaussian_Distribution_Setting(int layer)
        {
            gd[layer].input_unit = input_unit;
            gd[layer].output_unit = input_unit / 2;
            gd[layer].batch_sample = batch_sample;
            gd[layer].Setting();
            input_channel = 0;
            input_x = 0;
            input_y = 0;
            input_unit = gd[layer].output_unit;
            MaxUnit(gd[layer].input_unit, gd[layer].output_unit);
        }

        private void Activation_Setting(int layer)
        {
            activation_process[layer] = parameter[0];
            ac[layer].input_unit = input_unit;
            ac[layer].output_unit = input_unit;
            ac[layer].batch_sample = batch_sample;
            ac[layer].Setting();
            MaxUnit(ac[layer].input_unit, ac[layer].output_unit);
        }

        private void SaveParameter()
        {
            for (int i = 0; i < cp_stock; i++)
            {
                cp[i].SaveParameter(i);
            }

            for (int i = 0; i < dp_stock; i++)
            {
                dp[i].SaveParameter(i);
            }

            for (int i = 0; i < ct_stock; i++)
            {
                ct[i].SaveParameter(i);
            }
        }

        private void MaxUnit(int unit1, int unit2)
        {
            if (max_unit < unit1)
            {
                max_unit = unit1;
            }

            if (max_unit < unit2)
            {
                max_unit = unit2;
            }
        }

        private void Summary()
        {
            int cp_step = 0;
            int ct_step = 0;
            int pp_step = 0;
            int np_step = 0;
            int dp_step = 0;
            int gd_step = 0;

            for (int i = 0; i < sq_stock; i++)
            {
                if (sequence[i] == "conv2d")
                {
                    Console.WriteLine("conv2d");
                    Console.WriteLine("Input:(" + cp[cp_step].input_channel + "," + cp[cp_step].input_x + "," + cp[cp_step].input_y + ")");
                    Console.WriteLine("Filter:(" + cp[cp_step].filter_channel + "," + cp[cp_step].filter_x + "," + cp[cp_step].filter_y + ")");
                    Console.WriteLine("Output:(" + cp[cp_step].output_channel + "," + cp[cp_step].output_x + "," + cp[cp_step].output_y + ")");
                    Console.WriteLine("******************************************************************************************************");
                    cp_step += 1;

                }
                else if (sequence[i] == "conv2d_t")
                {
                    Console.WriteLine("conv2d transpose");
                    Console.WriteLine("Input:(" + ct[ct_step].input_channel + "," + ct[ct_step].input_x + "," + ct[ct_step].input_y + ")");
                    Console.WriteLine("Filter:(" + ct[ct_step].filter_channel + "," + ct[ct_step].filter_x + "," + ct[ct_step].filter_y + ")");
                    Console.WriteLine("Output:(" + ct[ct_step].output_channel + "," + ct[ct_step].output_x + "," + ct[ct_step].output_y + ")");
                    Console.WriteLine("******************************************************************************************************");
                    ct_step += 1;
                }
                else if (sequence[i] == "pooling")
                {
                    Console.WriteLine("pooling");
                    Console.WriteLine("Input:(" + pp[pp_step].input_channel + "," + pp[pp_step].input_x + "," + pp[pp_step].input_y + ")");
                    Console.WriteLine("Filter:(" + pp[pp_step].size_x + "," + pp[pp_step].size_y + ")");
                    Console.WriteLine("Output:(" + pp[pp_step].output_channel + "," + pp[pp_step].output_x + "," + pp[pp_step].output_y + ")");
                    Console.WriteLine("******************************************************************************************************");
                    pp_step += 1;
                }
                else if (sequence[i] == "norm")
                {
                    Console.WriteLine("normalization");
                    Console.WriteLine("Input:(" + np[np_step].input_channel + "," + np[np_step].input_x + "," + np[np_step].input_y + ")");
                    Console.WriteLine("Filter:(" + np[np_step].size_x + "," + np[np_step].size_y + ")");
                    Console.WriteLine("Output:(" + np[np_step].output_channel + "," + np[np_step].input_x + "," + np[np_step].input_y + ")");
                    Console.WriteLine("******************************************************************************************************");
                    np_step += 1;
                }
                else if (sequence[i] == "dense")
                {
                    Console.WriteLine("dense");
                    Console.WriteLine("Input:" + dp[dp_step].input_unit);
                    Console.WriteLine("Output:" + dp[dp_step].output_unit);
                    Console.WriteLine("******************************************************************************************************");
                    dp_step += 1;
                }
                else if (sequence[i] == "gaudis")
                {
                    Console.WriteLine("gaussain distribution");
                    Console.WriteLine("Input:" + gd[gd_step].input_unit);
                    Console.WriteLine("Output:" + gd[gd_step].output_unit);
                    Console.WriteLine("******************************************************************************************************");
                    gd_step += 1;
                }
            }
        }
    }
}