#Author: Pan Yue

library(zoo)
library(xts)
library(TTR)
library(gftUtils)

if (isDebugMode == TRUE) {
    save.image("/home/guofu/SIMPLE_SIMULATE_DAILY_TRADE_CHN_STK.RData") 
}

addMissingColumns <- function(m, columns, fillvalue=0) {
    missingColumns <- setdiff(columns, colnames(m))
    result <- m
    if (length(missingColumns)>0) {
        result <- cbind(result, matrix(fillvalue, nrow=nrow(m), ncol=length(missingColumns), \
		dimnames=list(rownames(m), missingColumns)))
    }
    return(result)
}

fillHolding <- function(d, nextd, tradeDates, holding, holdingCash, adjustFactor) {
    result <- list()
    if (nextd > d) {
        holdingDates <- tradeDates[tradeDates>=d & tradeDates<=nextd]
        holding[holdingDates, ] <- rep(holding[d, ], each = length(holdingDates))
        holdingCash[holdingDates, ] <- rep(holdingCash[d, ], each = length(holdingDates))
        holding[holdingDates, ] <- util.fill(holding[holdingDates, , drop=FALSE] * sweep(adjustFactor[holdingDates, , drop=FALSE], MARGIN=2, adjustFactor[holdingDates[1], ], "/"), 0)                
    }
    result$holding <- holding
    result$holdingCash <- holdingCash
    return(result)
}


#markToMarketPrice / totalReturnFactor / executePrice should be resample to business daily, if we want to run monthly strategy, please resample these price to monthly
#parameters: riseLimitThres, fallLimitThres, volumeLimitPct, canTradeOnSuspend, buyCommission, sellCommission, shiftBeginDateToSignal, execDelayPeriods, lotSize
#assume the dates in targetPortfolio and initialHolding should be business days!!!
#initialHolding should be in share
#execPriceReturn should be pass in additionalTs for monthly strategy
SIMPLE_SIMULATE_DAILY_TRADE_CHN_STK <- function(beginDate, endDate, initialHolding, targetPortfolioWgt, markToMarketPrice, totalReturnFactor, executePrice, execPriceReturn, tradeVolume, tradingParam, additionalTs){
    result <- list(error="", value=NA)
    cashGid <- util.getCashGid()
    
    if(!is.null(tradingParam$debug)){
        if(tradingParam$debug == 1){
            cat("Start the SIMPLE_SIMULATE_DAILY_TRADE_CHN_STK \n")	
        }
    }
    
    beginDate <- util.asdate(beginDate)
    endDate <- util.asdate(endDate)
    allDates <- sort(unique(intersect(intersect(intersect(intersect(rownames(markToMarketPrice), rownames(totalReturnFactor)), rownames(executePrice)), rownames(tradeVolume)), rownames(execPriceReturn))))    
    rebDates <- character(0)
    if (nrow(targetPortfolioWgt) > 0) {
        targetPortfolioWgt <- targetPortfolioWgt[rownames(targetPortfolioWgt) %in% allDates, ]  #get the targetportfolio on trading dates
        rebDates <- names(which(apply(!is.na(targetPortfolioWgt), 1, any)))  #remove the line with all NA values
        if (length(rebDates)>0 & sum(tradingParam$execDelayPeriods>0)>0) { #shift the rebalance date
            rebDates <- as.character(util.alignDate(util.asdate(rebDates) + tradingParam$execDelayPeriods, util.asdate(allDates)))
            targetPortfolioWgt <- targetPortfolioWgt[!is.na(rebDates),]
            rebDates <- rebDates[!is.na(rebDates)]
            rownames(targetPortfolioWgt) <- rebDates
        }        
    }
    
    holdingSymbols <- character(0)
    if (is.matrix(initialHolding))  {
        beginDate <- util.asdate(tail(rownames(initialHolding), 1))
        holdingSymbols <- sort(unique(setdiff(colnames(initialHolding), cashGid)))
    } else {
        if (sum(tradingParam$shiftBeginDateToSignal>0)>0 & nrow(targetPortfolioWgt) > 0) {
            beginDate <- max(beginDate, util.asdate(rownames(targetPortfolioWgt)[1]))
        }
    }

    tradeDates <- allDates[allDates >= beginDate & allDates <= endDate]
    beginDate <- util.asdate(tradeDates[1])
    endDate <- util.asdate(tail(tradeDates, 1))
    rebDates <- rebDates[rebDates >= beginDate & rebDates <= endDate]
    
    if(beginDate > endDate){
        result$error <- "Begin date must be less than end date!"
        return(result)
    }  
    
    allSymbols <- sort(unique(intersect(intersect(intersect(colnames(markToMarketPrice), colnames(executePrice)), colnames(tradeVolume)), colnames(execPriceReturn))))
    portfolioSymbols <- sort(unique(setdiff(colnames(targetPortfolioWgt), c(names(which(apply(is.na(targetPortfolioWgt) | !abs(targetPortfolioWgt)>0, 2, all))), cashGid))))
    if(sum(!holdingSymbols %in% allSymbols) > 0){
        result$error <- paste("Initial Portfolio has non A-share stocks!", holdingSymbols[!holdingSymbols %in% allSymbols][1])
        return(result)
    }
    if(sum(!portfolioSymbols %in% allSymbols) > 0){
        result$error <- paste("Target Portfolio has non A-share stocks! ", portfolioSymbols[!portfolioSymbols %in% allSymbols][1])
        return(result)
    }
    
    allSymbols <- sort(unique(setdiff(intersect(allSymbols, c(holdingSymbols, portfolioSymbols)), cashGid)))
    
    priceDates <- allDates[allDates >= beginDate-20 & allDates <= endDate]
    markToMarketPrice <- markToMarketPrice[priceDates, allSymbols]
    totalReturnFactor <- addMissingColumns(totalReturnFactor, allSymbols, 1)[priceDates, allSymbols]
    executePrice <- executePrice[priceDates, allSymbols]
    execPriceReturn <- execPriceReturn[priceDates, allSymbols]    
    tradeVolume <- tradeVolume[priceDates, allSymbols]
    
    if (!is.matrix(initialHolding)) {
        initialHolding <- matrix(initialHolding, nrow = 1, ncol = 1)
        rownames(initialHolding) <- as.character(beginDate)
        colnames(initialHolding) <- cashGid
    }
    initialHolding <- addMissingColumns(initialHolding, c(allSymbols,cashGid))
    initialHolding <- util.fill(tail(initialHolding, 1),0)
    rownames(initialHolding) <- as.character(beginDate)
    initialHoldingCash <- initialHolding[, cashGid, drop=FALSE]
    initialHolding <- initialHolding[, allSymbols, drop=FALSE]
    initialHoldingValue <- as.numeric(sum(initialHolding * markToMarketPrice[rownames(initialHolding), ],na.rm=TRUE) + initialHoldingCash)
    
    targetPortfolioWgt <- util.fill(addMissingColumns(targetPortfolioWgt, c(allSymbols,cashGid)),0)[rebDates, , drop=FALSE]
    if (any(targetPortfolioWgt<0)) {
        result$error <- "Do not support stock short selling and cash borrowing"
        return(result)        
    }
    targetPortfolioCashWgt <- targetPortfolioWgt[, cashGid, drop=FALSE]
    targetPortfolioWgt <- targetPortfolioWgt[, allSymbols, drop=FALSE]
    targetPortfolioCashWgt[,] <- 1 - apply(targetPortfolioWgt,1,sum, na.rm=TRUE)  #assume the portfolio contains stock and cash only
    
    buyVolume <- util.fill(tradeVolume, 0)
    sellVolume <- util.fill(tradeVolume, 0)
    if (sum(tradingParam$canTradeOnSuspend>0)>0) {
        buyVolume[buyVolume<1] <- Inf
        sellVolume[sellVolume<1] <- Inf        
    }
    if (sum(tradingParam$riseLimitThres>0)>0) {
        riseLimit <- execPriceReturn > tradingParam$riseLimitThres
        buyVolume[riseLimit] <- 0
        sellVolume[riseLimit & sellVolume>0] <- Inf
    }
    if (sum(tradingParam$fallLimitThres<0)>0) {
        fallLimit <- execPriceReturn < tradingParam$fallLimitThres
        buyVolume[fallLimit & buyVolume>0] <- Inf
        sellVolume[fallLimit] <- 0
    }
    if (sum(tradingParam$volumeLimitPct>0)>0) {
        buyVolume <- buyVolume * tradingParam$volumeLimitPct
        sellVolume <- sellVolume * tradingParam$volumeLimitPct
    } else {
        buyVolume[buyVolume>0] <- Inf
        sellVolume[sellVolume>0] <- Inf
    }
    lotSize <- util.getParm(tradingParam, "lotSize", 0)
    buyVolume <- util.round2lot(buyVolume, lotSize) #round to lotsize
    sellVolume <- util.round2lot(sellVolume, lotSize)
    
    buyCommission <- util.getParm(tradingParam, "buyCommission", 0)
    sellCommission <- util.getParm(tradingParam, "sellCommission", 0)

    holding <- matrix(0, nrow=length(tradeDates), ncol=length(allSymbols), dimnames=list(tradeDates, allSymbols))
    weights <- holding
    execution <- holding
    holdingCash <- matrix(0, nrow=length(tradeDates), ncol=1, dimnames=list(tradeDates, cashGid))
    portfolioValue <- matrix(0, nrow=length(tradeDates), ncol=1, dimnames=list(tradeDates, util.getGodGid()))
    cumRets <- portfolioValue
    singlePeriodRets <- portfolioValue
    turnoverPct <- portfolioValue

    d <- tradeDates[1]
    holding[d, ] <- initialHolding
    holdingCash[d,] <- initialHoldingCash    
    if (length(rebDates) < 1) { #no rebalance needed, so keep the inital holding
        nextd <- tail(tradeDates, 1)
        adjustedHoldings <- fillHolding(d, nextd, tradeDates, holding, holdingCash, totalReturnFactor)
        holding <- adjustedHoldings$holding
        holdingCash <- adjustedHoldings$holdingCash        
    } else {
        #fill the holding before the first rebalance
        nextd <- rebDates[1]
        adjustedHoldings <- fillHolding(d, nextd, tradeDates, holding, holdingCash, totalReturnFactor)
        holding <- adjustedHoldings$holding
        holdingCash <- adjustedHoldings$holdingCash                    
        
        for (i in 1:length(rebDates)) {
            d <- rebDates[i]
            currentHoldingValue <- util.fill(holding[d, , drop=FALSE] * executePrice[d, , drop=FALSE], 0)
            totalValue <- sum(currentHoldingValue, na.rm=TRUE) + holdingCash[d, ]
            currentHoldingWgt <- currentHoldingValue / totalValue
            currentHoldingCashWgt <- 1 - sum(currentHoldingWgt, na.rm=TRUE)
            targetHoldingWgt <- targetPortfolioWgt[d, , drop=FALSE]
            targetHoldingCashWgt <- 1 - sum(targetHoldingWgt, na.rm=TRUE)
            orderWgt <- targetHoldingWgt - currentHoldingWgt
            sellOrderWgt <- ifelse(orderWgt<0, orderWgt, 0)
            buyOrderWgt <- ifelse(orderWgt>0, orderWgt, 0)
            cashAvail <- holdingCash[d, ]
            if (sum(sellOrderWgt, na.rm=TRUE) < 0) { #need to sell
                sellOrder <- util.round2lot(sellOrderWgt / ifelse(currentHoldingWgt>0, currentHoldingWgt, 1) * holding[d, ], lotSize)
                sellOrder <- ifelse(targetHoldingWgt>0, sellOrder, -holding[d, ])   #sell all the stock if the target holding weight is zero, even if the share can not be round to lotSize
                sellExecution <- sellOrder
                sellExecution[,] <- -pmin(abs(sellExecution), sellVolume[d,])
                cashAvail <- cashAvail + sum(abs(sellExecution) * executePrice[d, ], na.rm=TRUE) * (1-sellCommission)
                execution[d,] <- execution[d,] + sellExecution
                holding[d, ] <- holding[d, ] + sellExecution
            }
            if (sum(buyOrderWgt, na.rm=TRUE) > 0) { #need to buy
                canBuyWgt <- cashAvail/totalValue - targetHoldingCashWgt
                if (canBuyWgt > 0) {  #cash available to buy stocks
                    buyOrder <- util.round2lot(util.fill(min(canBuyWgt/sum(buyOrderWgt, na.rm=TRUE), 1) * buyOrderWgt * totalValue / (1+buyCommission) / executePrice[d, ], 0), lotSize)
                    buyExecution <- buyOrder
                    buyExecution[,] <- pmin(buyExecution, buyVolume[d, ])
                    cashAvail <- cashAvail - sum(abs(buyExecution) * executePrice[d, ], na.rm=TRUE) * (1+buyCommission)
                    execution[d,] <- execution[d,] + buyExecution
                    holding[d, ] <- holding[d, ] + buyExecution
                }
            }
            holdingCash[d,] <- cashAvail
            turnoverPct[d,] <- sum(abs(execution[d, ]) * executePrice[d, ], na.rm=TRUE) / totalValue
            
            if (i < length(rebDates)) {  #fill the holding from current rebalance to the next
                nextd <- rebDates[i+1]
                adjustedHoldings <- fillHolding(d, nextd, tradeDates, holding, holdingCash, totalReturnFactor)
                holding <- adjustedHoldings$holding
                holdingCash <- adjustedHoldings$holdingCash                                    
            }
        }
        #fill the holding after the last rebalance        
        nextd <- tail(tradeDates, 1)
        adjustedHoldings <- fillHolding(d, nextd, tradeDates, holding, holdingCash, totalReturnFactor)
        holding <- adjustedHoldings$holding
        holdingCash <- adjustedHoldings$holdingCash                    
    }
    portfolioValue[,] <- apply(holding * markToMarketPrice[tradeDates, ], 1, sum, na.rm=TRUE) + holdingCash
    weights <- sweep(holding * markToMarketPrice[tradeDates, ], MARGIN=1, portfolioValue, FUN="/")
    cumRets <- portfolioValue / initialHoldingValue - 1
    singlePeriodRets <- portfolioValue / util.lag(portfolioValue, n=1, fillvalue=initialHoldingValue) - 1

    result$value <- list()
    result$value$HOLDING <- cbind(util.replaceZero(holding, NA), holdingCash)
    result$value$WEIGHTS <- util.replaceZero(weights, NA)
    result$value$CUMULATIVE_RETURN <- cumRets
    result$value$SINGLE_PERIOD_RETURN <- singlePeriodRets
    result$value$PORTFOLIO_VALUE <- portfolioValue
    result$value$TURNOVER <- turnoverPct
    return(result)
}
