import torch
import plotly.graph_objects as go


class sigmoid_approx(torch.nn.Module):
    def __init__(self, shift, slope=1.0,min_val=0.0,max_val=1.0):
        super(sigmoid_approx, self).__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.shift = shift
        self.slope = slope

    def forward(self, input):
        result = input * self.slope
        result = result + self.shift
        result = torch.nn.functional.hardtanh(result, self.min_val, self.max_val, False)
        return result


if __name__ == '__main__':
    input = torch.linspace(-10, 10)
    sig = torch.nn.Sigmoid()
    under_approx = sigmoid_approx(0, slope = 0.2, max_val=0.99)
    over_approx = sigmoid_approx(1, slope = 0.2, min_val=0.01)
    result_under = under_approx(input)
    result_over = over_approx(input)
    result_sigmoid = sig(input)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=input.numpy(), y=result_under.numpy(), mode='lines+markers', name='under_approx'))
    fig.add_trace(go.Scatter(x=input.numpy(), y=result_over.numpy(), mode='lines+markers', name='over_approx'))
    fig.add_trace(go.Scatter(x=input.numpy(), y=result_sigmoid.numpy(), mode='lines+markers', name='sigmoid'))
    fig.show()
