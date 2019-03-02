# Forecasting_RealEstate-
Using time series for forecasting two-bedroom prices in Orange County 

Orange County California is a haven for surfers, high-income families, and immigrants from Mexico and Asia. The OC is widespread consisting of 9 cities including Anaheim,  Newport Beach, and Santa Ana.

Like most California counties, Orange County home prices survived the financial crash and are now some of the most valuable real estate in the states. For reference, Orange County homes are worth around $111,000 more than the average U.S. county (227 counties).[1]

Our goal is to predict future Orange County real estate prices. As a starter, we will build multiple prediction models based on varying time series lags. In other words, we are shifting our time series model n steps (picking different values of n) back in reference to current time t.

We will train the model picking random years (between 1996-2018) and determine which lag (t-n) is the best predictor of the actual real estate price of two bedroom homes in Orange County via model coefficients.

The time series model with lag 6 gives us the lowest mean squared error so we will use this model to forecast two-bedroom house prices.

Read Here: https://beyondtheaverage.wordpress.com/2019/02/26/orange-county-real-estate-is-a-smart-investment-if-you-can-afford-it/
