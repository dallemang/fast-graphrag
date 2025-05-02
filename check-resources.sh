#!/bin/bash
# check-resources.sh - Check current resources and highlight costly ones

# App name
APP=kgc-ddw-entity

echo "==========================================="
echo "HEROKU RESOURCE STATUS CHECK"
echo "App: $APP"
echo "Time: $(date)"
echo "==========================================="

# Check dyno types and count
echo -e "\n📊 DYNO INFORMATION:"
heroku ps -a $APP

# Extract dyno types for analysis
WEB_TYPE=$(heroku ps -a $APP | grep "web." | awk '{print $2}')
WORKER_TYPE=$(heroku ps -a $APP | grep "worker." | awk '{print $2}')
if [[ -z "$WORKER_TYPE" ]]; then
  # If worker is crashed or not found, try to get its type another way
  WORKER_TYPE=$(heroku ps:type -a $APP | grep "worker" | awk '{print $3}')
fi
WORKER_COUNT=$(heroku ps -a $APP | grep "worker." | wc -l)
# Ensure we count at least 1 worker if it exists but is crashed
if [[ $WORKER_COUNT -eq 0 && ! -z "$WORKER_TYPE" ]]; then
  WORKER_COUNT=1
fi

# Check add-ons
echo -e "\n🧩 ADD-ONS:"
heroku addons -a $APP

# Extract Redis info
REDIS_INFO=$(heroku addons -a $APP | grep -i "redis")

# Analysis & Cost Estimates
echo -e "\n💰 COST ANALYSIS:"
echo "-------------------------------------------"

TOTAL_COST=0
HIGH_COST=false

# Web dyno cost analysis
if [[ "$WEB_TYPE" == "Standard-1X" ]]; then
  echo "⚠️  Web dyno: Standard-1X ($0.05/hour)"
  TOTAL_COST=$(echo "$TOTAL_COST + 0.05" | bc)
  HIGH_COST=true
elif [[ "$WEB_TYPE" == "Standard-2X" ]]; then
  echo "⚠️  Web dyno: Standard-2X ($0.10/hour)"
  TOTAL_COST=$(echo "$TOTAL_COST + 0.10" | bc)
  HIGH_COST=true
elif [[ "$WEB_TYPE" == "Performance"* ]]; then
  echo "🔴 Web dyno: $WEB_TYPE (EXPENSIVE: $0.30+/hour)"
  TOTAL_COST=$(echo "$TOTAL_COST + 0.30" | bc)
  HIGH_COST=true
else
  echo "✅ Web dyno: Free or Hobby ($0.007/hour)"
  TOTAL_COST=$(echo "$TOTAL_COST + 0.007" | bc)
fi

# Worker dyno cost analysis
if [[ "$WORKER_TYPE" == "Standard-1X" ]]; then
  WORKER_COST=$(echo "0.05 * $WORKER_COUNT" | bc)
  echo "⚠️  Worker dynos: $WORKER_COUNT x Standard-1X ($WORKER_COST/hour)"
  TOTAL_COST=$(echo "$TOTAL_COST + $WORKER_COST" | bc)
  HIGH_COST=true
elif [[ "$WORKER_TYPE" == "Standard-2X" ]]; then
  WORKER_COST=$(echo "0.10 * $WORKER_COUNT" | bc)
  echo "⚠️  Worker dynos: $WORKER_COUNT x Standard-2X ($WORKER_COST/hour)"
  TOTAL_COST=$(echo "$TOTAL_COST + $WORKER_COST" | bc)
  HIGH_COST=true
elif [[ "$WORKER_TYPE" == "Performance"* ]]; then
  WORKER_COST=$(echo "0.30 * $WORKER_COUNT" | bc)
  echo "🔴 Worker dynos: $WORKER_COUNT x $WORKER_TYPE (EXPENSIVE: $WORKER_COST+/hour)"
  TOTAL_COST=$(echo "$TOTAL_COST + $WORKER_COST" | bc)
  HIGH_COST=true
elif [[ $WORKER_COUNT -gt 1 ]]; then
  WORKER_COST=$(echo "0.007 * $WORKER_COUNT" | bc)
  echo "⚠️  Worker dynos: $WORKER_COUNT x Hobby ($WORKER_COST/hour)"
  TOTAL_COST=$(echo "$TOTAL_COST + $WORKER_COST" | bc)
  HIGH_COST=true
else
  echo "✅ Worker dyno: Free or Hobby ($0.007/hour)"
  TOTAL_COST=$(echo "$TOTAL_COST + 0.007" | bc)
fi

# Redis analysis
if [[ "$REDIS_INFO" == *"premium"* ]]; then
  echo "🔴 Redis: Premium plan (EXPENSIVE: $0.60+/hour)"
  TOTAL_COST=$(echo "$TOTAL_COST + 0.60" | bc)
  HIGH_COST=true
elif [[ "$REDIS_INFO" == *"mini"* ]]; then
  echo "✅ Redis: Mini plan ($0.021/hour)"
  TOTAL_COST=$(echo "$TOTAL_COST + 0.021" | bc)
elif [[ "$REDIS_INFO" == *"heroku-redis"* ]]; then
  echo "⚠️ Redis: Unknown Heroku Redis plan"
  TOTAL_COST=$(echo "$TOTAL_COST + 0.021" | bc)
elif [[ "$REDIS_INFO" == *"upstash"* ]]; then
  echo "✅ Redis: Upstash (pay per use)"
else
  echo "🟢 No Redis detected"
fi

# Overall assessment
echo "-------------------------------------------"
printf "Estimated hourly cost: \$%.2f\n" $TOTAL_COST
DAILY_COST=$(echo "$TOTAL_COST * 24" | bc)
printf "If left running 24 hours: \$%.2f per day\n" $DAILY_COST

if $HIGH_COST; then
  echo -e "\n🚨 WARNING: Expensive resources detected!"
  echo "Consider running the scale-down script if the workshop is over:"
  echo "./scale-down.sh"
else
  echo -e "\n✅ No expensive resources detected. Using budget-friendly configuration."
fi

echo "==========================================="
 No newline at end of file
