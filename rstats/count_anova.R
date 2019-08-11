# Basic statistical tests for paper
# have run the tests with data points being each day, the monthly means
# and the yearly means. The yearly means is the correct one to use in my
# case I think as it smooths out the seasonal component.
library(ggplot2)
library(lubridate)
library(lsmeans)

# Setup
setwd("~/Development/PhD/repos/oceancc")

# get data
df_Jerv <- read.csv('./data/count_Jerv.csv')
df_Jerv$site <- 'Jerv'
df_Bate <-
read.csv('./data/count_Bate.csv')
df_Bate$site <- 'Bate'
df_Howe <- read.csv('./data/count_Howe.csv')
df_Howe$site <- 'Howe'

# bind data
df <- do.call('rbind', list(df_Jerv, df_Bate, df_Howe))
# convert string to date
df$dt <- as.POSIXct(df$dt)
# convert string to factor
df$site <- as.factor(df$site)
# make Ratio data
df$ratioA = df$countA/(df$countA + df$countB)
df$ratioB = df$countB/(df$countA + df$countB)
# add year and month
df$year <- year(df$dt)
df$month <- month(df$dt)

# get monthly and yearly means
i <- 1
df_ls <- split(df, df$site)
df_year <- list()
df_month <- list()
for (dat in df_ls){
  df_year[[i]] <- aggregate(dat, by=list(dat$year), mean)
  df_month[[i]] <- aggregate(dat, by=list(dat$year, dat$month), mean) 
  df_year[[i]]$site <- dat$site[1]
  df_month[[i]]$site <- dat$site[1]
  i <- i +1
}
df_year <- do.call('rbind', df_year)
df_month <- do.call('rbind',df_month)
# drop 2016 in years
df_year <- df_year[!df_year$year == 2016,]

# plots
ggplot(df, aes(x=dt, y=ratioA, colour=site)) +
  geom_point(alpha=0.4) +
  stat_smooth(data=subset(df, site == "Jerv"), method = "lm") +
  stat_smooth(data=subset(df, site == "Bate"), method = "lm") +
  stat_smooth(data=subset(df, site == "Howe"), method = "lm") +
  ggtitle('Days')

ggplot(df_month, aes(x=dt, y=ratioA, colour=site)) +
  geom_point(alpha=0.4) +
  stat_smooth(data=subset(df, site == "Jerv"), method = "lm") +
  stat_smooth(data=subset(df, site == "Bate"), method = "lm") +
  stat_smooth(data=subset(df, site == "Howe"), method = "lm") +
  ggtitle('Months')

ggplot(df_year, aes(x=dt, y=ratioA, colour=site)) +
  geom_point(alpha=0.4) +
  stat_smooth(data=subset(df, site == "Jerv"), method = "lm") +
  stat_smooth(data=subset(df, site == "Bate"), method = "lm") +
  stat_smooth(data=subset(df, site == "Howe"), method = "lm") +
  ggtitle('Years')

# make models
print('\nModel all days')
fit <- lm(formula = ratioA ~ dt*site, data = df)
anova(fit)
TukeyHSD(aov(fit))

print('\nModel all month means')
fit <- lm(formula = ratioA ~ dt*site, data = df_month)
anova(fit)
TukeyHSD(aov(fit))

print('\nModel all year means')
fit <- lm(formula = ratioA ~ dt*site, data = df_year)
anova(fit)
TukeyHSD(aov(fit))

