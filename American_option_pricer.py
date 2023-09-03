import torch
import numpy as np
import warnings
import time
from sklearn.linear_model import LinearRegression
from torch.autograd import grad




class AmericanOptionsLSMC(object):
    """ Class for American put options pricing using LSMC
    S0 : float : initial stock/index level
    strike : float : strike price
    T : float : time to maturity (in year fractions)
    M : int : grid or granularity for time (in number of total points)
    r : float : constant risk-free short rate
    div :    float : dividend yield
    sigma :  float : volatility factor in diffusion term
    simulations: int: need to be even number
    """

    def __init__(self, type, S, K, T, M, r, div, sigma, func_num, n_poly, simulations, seed):
        try:
            self.type = type
            self.S = torch.tensor(S, requires_grad=True)
            self.K = torch.tensor(K)
            assert T > 0
            self.T = torch.tensor(T, requires_grad=True)
            assert M > 0
            self.M = int(M)
            assert r >= 0
            self.r = torch.tensor(r, requires_grad=True)
            assert div >= 0
            self.div = torch.tensor(div, requires_grad=True)
            assert sigma > 0
            self.sigma = torch.tensor(sigma, requires_grad=True)
            assert func_num > 0
            self.func_num = func_num
            assert n_poly > 0
            self.n_poly = int(n_poly)
            assert simulations > 0
            self.simulations = int(simulations)
            assert seed > 0
            self.seed = int(seed)
        except ValueError:
            print('Error passing Put Options parameters')

        if S < 0 or K < 0 or T <= 0 or r < 0 or sigma < 0:
            raise ValueError('Error: Negative inputs not allowed')

        self.time_unit = self.T / float(self.M)
        self.discount = torch.exp(-(self.r - self.div) * self.time_unit)

        self.func_dic = {1: self.Poly_reg_filter, 2: self.Poly_reg_no_filter,
                         3: self.Poly_reg_choose_deg_filter, 4: self.Poly_reg_choose_deg_no_filter,
                         5: self.Laguerre_poly_reg_filter, 6: self.Laguerre_poly_reg_no_filter}
        self.func = self.func_dic[self.func_num]
        self.V0 = 0
        self.std = 0
        self.first_order_greeks = ()

    def make_features(self, x, n_poly):
        x = x.unsqueeze(1)
        return torch.cat([torch.pow(x, i) for i in range(n_poly + 1)], 1)

    def get_batch(self, data_x, data_y, n_poly, K):
        cond = torch.where(data_x > K)
        x = data_x[cond]
        x = self.make_features(x, n_poly)
        y = data_y.unsqueeze(1)[cond]
        return x, y
        # if torch.cuda.is_available():
        #     return Variable(x).cuda(), Variable(data_y).cuda()
        # else:
        #     return Variable(x), Variable(data_y)

    def filter_data(self, data_x, data_y, K):
        if self.type == 'C':
            cond = torch.where(data_x > K)
        else:
            cond = torch.where(data_x < K)
        x = data_x[cond]
        y = data_y[cond]
        return x, y, cond





    def Poly_reg_filter(self, data_x, data_y):
        x, y, cond = self.filter_data(data_x, data_y, self.K)
        x = x.unsqueeze(1)
        X = torch.cat([torch.pow(x, i) for i in range(1, self.n_poly + 1)], 1)
        X, Y = X.detach().numpy(), y.detach().numpy()
        try:
            regressor = LinearRegression()
            regressor.fit(X, Y * self.discount.detach().numpy())
        except ValueError:
            raise Exception("At the time point (node) all paths are not optimal to convert into shares")
        continuation_value = torch.from_numpy(regressor.predict(X))
        return continuation_value, cond

    def Poly_reg_no_filter(self, data_x, data_y):
        X, Y = data_x.detach().numpy(), data_y.detach().numpy()
        regression = np.polynomial.polynomial.polyfit(X,
                                                      Y * self.discount.detach().numpy(), self.n_poly)
        x_poly = self.make_features(data_x, self.n_poly)
        continuation_value = torch.sum(torch.tensor(regression) * x_poly, dim=1)
        return continuation_value

    def Poly_reg_choose_deg_filter(self, data_x, data_y):
        x, y, cond = self.filter_data(data_x, data_y, self.K)
        X, Y = x.detach().numpy(), y.detach().numpy()
        try:
            for deg in np.arange(1, 6, 1):
                with warnings.catch_warnings():
                    warnings.filterwarnings('error')
                    try:
                        regression = np.polyfit(X,
                                                Y * self.discount.detach().numpy(), deg)
                    except np.RankWarning:
                        break
        except ValueError:
            raise Exception("At the time point (node) all paths are not optimal to convert into shares")
        continuation_value = torch.from_numpy(np.polyval(regression, X))
        return continuation_value, cond

    def Poly_reg_choose_deg_no_filter(self, data_x, data_y):
        X, Y = data_x.detach().numpy(), data_y.detach().numpy()
        for deg in np.arange(1, 6, 1):
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    regression = np.polyfit(X,
                                            Y * self.discount.detach().numpy(), deg)
                except np.RankWarning:
                    break
        continuation_value = torch.from_numpy(np.polyval(regression, X))
        return continuation_value

    def Laguerre_poly_reg_filter(self, data_x, data_y):
        x, y, cond = self.filter_data(data_x, data_y, self.K)
        x = x.unsqueeze(1)
        x = torch.cat([torch.exp(-x / 2), torch.exp(-x / 2) * (1 - x),
                       torch.exp(-x / 2) * (1 - 2 * x + x ** 2 / 2)], 1)
        X, Y = x.detach().numpy(), y.detach().numpy()
        try:
            regressor = LinearRegression()
            regressor.fit(X, Y * self.discount.detach().numpy())
        except ValueError:
            raise Exception("At the time point (node) all paths are not optimal to convert into shares")
        continuation_value = torch.from_numpy(regressor.predict(X))
        return continuation_value, cond

    def Laguerre_poly_reg_no_filter(self, data_x, data_y):
        x = data_x.unsqueeze(1)
        x = torch.cat([torch.exp(-x / 2), torch.exp(-x / 2) * (1 - x),
                       torch.exp(-x / 2) * (1 - 2 * x + x ** 2 / 2)], 1)
        X, Y = x.detach().numpy(), data_y.detach().numpy()
        try:
            regressor = LinearRegression()
            regressor.fit(X, Y * self.discount.detach().numpy())
        except ValueError:
            raise Exception("At the time point (node) all paths are not optimal to convert into shares")
        continuation_value = torch.from_numpy(regressor.predict(X))
        return continuation_value





   # @property
    def price(self):
        '''This part calculates option price and calculated price std'''

        '''initialize first time price'''
        init = self.S
        first = self.S * torch.ones(self.simulations)
        prices = []
        prices.append(first.unsqueeze(0))
        #        torch.manual_seed(self.seed)

        for j in range(1, int(self.T * self.M) + 1):
            '''antithetic for brownian motion'''
            np.random.seed(self.seed + j)
            brownian = np.random.standard_normal(self.simulations // 2)
            brownian = torch.tensor(np.concatenate((brownian, -brownian)))
            '''get underlying price metrix'''
            W_T = torch.sqrt(torch.tensor(1. / self.M)) * brownian
            mult = torch.exp(
                (self.r - self.div - 0.5 * torch.square(self.sigma)) * self.time_unit + self.sigma * W_T)
            init = init * mult
            prices.append(init.unsqueeze(0))
        MCprice_matrix = torch.cat(prices, dim=0)

        '''get payoff matrix'''
        if self.type == 'C':
            payoff = torch.maximum(MCprice_matrix - self.K,
                               torch.zeros([int(self.M * self.T) + 1, self.simulations]))
        else:
            payoff = torch.maximum(self.K - MCprice_matrix,
                                   torch.zeros([int(self.M * self.T) + 1, self.simulations]))

        '''conduct backward pricing'''
        for i in range(int(self.M * self.T) - 1, 0, -1):
            if i == int(self.M * self.T) - 1:
                V_pre = payoff[i + 1]

            if self.func_num % 2 == 1:
                continuation_value, cond = self.func(MCprice_matrix[i], V_pre)
                V = self.discount * V_pre
                V[cond] = torch.where(continuation_value > payoff[i][cond], V_pre[cond] * self.discount, payoff[i][cond])
            else:
                continuation_value = self.func(MCprice_matrix[i], V_pre)
                V = torch.where(continuation_value > payoff[i], V_pre * self.discount, payoff[i])  # calculate option value
#            V = torch.where(continuation_value > payoff[i], V_pre * self.discount, payoff[i])
#            V = torch.reshape(V, (1, -1))
            V_pre = V
        V0 = self.discount * torch.mean(V)
        # if self.type == 'C':
        #     V0 = torch.maximum(V0, self.S-self.K)
        # else:
        #     V0 = torch.maximum(V0, self.K-self.S)

        # return V0, std
        '''payoff0: option payoff if exercised initially'''
        if self.type == 'C':
            payoff0 = self.S - self.K
        else:
            payoff0 = self.K - self.S
        self.V0 = torch.maximum(V0, payoff0)  # get final option value
        self.std = self.discount * torch.std(V)  # calculate option value stability

    #    @property
    def calculate_greeks(self):
        self.V0.backward()

    @property
    def delta(self):
        return self.S.grad

    @property
    def vega(self):
        return self.sigma.grad

    @property
    def theta(self):
        diff = 1. / self.M
        call_1 = AmericanOptionsLSMC(self.type, np.float64(self.S),
                                     np.float64(self.K), np.float64(self.T + diff), self.M,
                                     np.float64(self.r), np.float64(self.div), np.float64(self.sigma),
                                     self.func_num,
                                     self.n_poly, self.simulations, self.seed)
        call_1.price()
        call_2 = AmericanOptionsLSMC(self.type, np.float64(self.S),
                                     np.float64(self.K), np.float64(self.T - diff), self.M,
                                     np.float64(self.r), np.float64(self.div), np.float64(self.sigma),
                                     self.func_num,
                                     self.n_poly, self.simulations, self.seed)
        call_2.price()
        return (call_1.V0 - call_2.V0) / float(2. * diff)

    @property
    def rho(self):
        return self.r.grad

    @property
    def phi(self):
        return self.div.grad

    @property
    def gamma_and_vanna(self):
        diff = self.S * 0.01
        call_1 = AmericanOptionsLSMC(self.type, np.float64(self.S + diff),
                                     np.float64(self.K), np.float64(self.T), self.M,
                                     np.float64(self.r), np.float64(self.div), np.float64(self.sigma),
                                     self.func_num,
                                     self.n_poly, self.simulations, self.seed)
        call_1.price()
        call_1.calculate_greeks()
        call_2 = AmericanOptionsLSMC(self.type, np.float64(self.S - diff),
                                     np.float64(self.K), np.float64(self.T), self.M,
                                     np.float64(self.r), np.float64(self.div), np.float64(self.sigma),
                                     self.func_num,
                                     self.n_poly, self.simulations, self.seed)
        call_2.price()
        call_2.calculate_greeks()
        return (call_1.delta - call_2.delta) / float(2. * diff), (call_1.vega - call_2.vega) / float(2. * diff)

    @property
    def volga(self):
        diff = self.sigma * 0.01
        call_1 = AmericanOptionsLSMC(self.type, np.float64(self.S),
                                     np.float64(self.K), np.float64(self.T), self.M,
                                     np.float64(self.r), np.float64(self.div), np.float64(self.sigma + diff),
                                     self.func_num,
                                     self.n_poly, self.simulations, self.seed)
        call_1.price()
        call_1.calculate_greeks()
        call_2 = AmericanOptionsLSMC(self.type, np.float64(self.S),
                                     np.float64(self.K), np.float64(self.T), self.M,
                                     np.float64(self.r), np.float64(self.div), np.float64(self.sigma - diff),
                                     self.func_num,
                                     self.n_poly, self.simulations, self.seed)
        call_2.price()
        call_2.calculate_greeks()
        return (call_1.vega - call_2.vega) / float(2. * diff)
    # def get_greeks(self):
    #     self.first_order_greeks = grad(self.V0, [self.S, self.sigma, self.r, self.T, self.div]
    #                                    , create_graph=True)
    #     self.gamma_and_vanna = grad(self.first_order_greeks[0], [self.S, self.sigma], retain_graph=True)
    #     self.volga = grad(self.first_order_greeks[1], self.sigma)



if __name__ == "__main__":

    ave = 0
    delta = gamma = vega = rho = theta = phi = volga = vanna = 0

    type = 'P'
    S = 36.
    K = 40.
    T = 1.
    M = 50
    r = 0.06
    div = 0.0
    sigma = 0.2
    n_poly = 2
    n_sim = 100000
    n_time = 1

    func_num = 3
    '''
    1: self.Poly_reg_filter
    2: self.Poly_reg_no_filter
    3: self.Poly_reg_choose_deg_filter
    4: self.Poly_reg_choose_deg_no_filter
    5: self.Laguerre_poly_reg_filter
    6: self.Laguerre_poly_reg_no_filter
    '''
    time1 = time2 = time3 = 0
    for n_t in range(n_time):
        start = time.time()
        AmericanOption = AmericanOptionsLSMC(type, S, K, T, M, r, div, sigma, func_num, n_poly, n_sim, 123)
        AmericanOption.price()
        start2 = time.time()
        AmericanOption.calculate_greeks()
        start3 = time.time()
        ave = (ave * n_t + float(AmericanOption.V0)) / (n_t + 1)
        delta = (delta * n_t + AmericanOption.delta) / (n_t + 1)
        vega = (vega * n_t + AmericanOption.vega) / (n_t+1)
        rho = (rho * n_t + AmericanOption.rho) / (n_t+1)
        theta = (theta * n_t + AmericanOption.theta) / (n_t+1)
        phi = (phi * n_t + AmericanOption.phi) / (n_t + 1)
        gamma_temp, vanna_temp = AmericanOption.gamma_and_vanna
        volga_temp = AmericanOption.volga
        gamma = (gamma * n_t + gamma_temp) / (n_t + 1)
        vanna = (vanna * n_t + vanna_temp) / (n_t + 1)
        volga = (volga * n_t + volga_temp) / (n_t + 1)
        end = time.time()
        time1 = (time1 * n_t + (start2 - start)) / (n_t + 1)
        time2 = (time2 * n_t + (start3 - start2)) / (n_t + 1)
        time3 = (time3 * n_t + (end - start3)) / (n_t + 1)
    print (f'Price: {ave}')

    print (f'Delta: {delta}')
    print (f'Gamma: {gamma}')
    print (f'Vega: {vega}')
    print (f'Rho: {rho}')
    print (f'Theta: {theta}')
    print(f'Phi: {phi}')
    print(f'Volga: {volga}')
    print(f'Vanna: {vanna}')
    print (f'std: {AmericanOption.std}')
    print(f'time1: {time1}')
    print(f'time2: {time2}')
    print(f'time3: {time3}')
    torch.cuda.empty_cache()
