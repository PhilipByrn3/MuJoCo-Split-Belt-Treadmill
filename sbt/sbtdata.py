import pandas as pd
import numpy as np
import math

def ConstructData(timedata, fprdat, fpbdat, measured_velocity):

    datapd = {

    'Time':timedata,
    'fprdat':fprdat,
    'fpbdat':fpbdat,
    'measured_velocity':measured_velocity,

    }

    df = pd.DataFrame(datapd)
    df['fprdat_shift'] = df['fprdat'].shift(-16)
    df['fpbdat_shift'] = df['fpbdat'].shift(-16)

    fprdat = df['fprdat']
    fpbdat = df['fpbdat']
    fprdat_shift = df['fprdat_shift']
    fpbdat_shift = df['fpbdat_shift']

    measured_velocity = df['measured_velocity']

    return [df, fprdat, fprdat_shift, fpbdat, fpbdat_shift, measured_velocity]



def findTime(df, fpdat, fpdat_shift, print_data):

    time_df = df['Time']
    idx = 1
    j = 0
    lookback = 0
    lookahead = 0
    total_difference = 0
    tavg = 0

    fpdat_shift = fpdat_shift.copy()
    fpdat = fpdat.copy()

    fpdat_shift[fpdat_shift == 98.10000000000001] = 0
    fpdat_shift[fpdat_shift != 0] = 1

    fpdat[fpdat==98.10000000000001] = 0
    fpdat[fpdat!=0] = 1
    start_times = []  # Stack to keep track of start times
    pairs = []  # List to store the pairs of start and end times

    for idx in range(len(fpdat_shift) - 16):

        while j < 16:
            lookback = lookback + fpdat[idx + j]
            lookahead = lookahead + fpdat_shift[idx + j]
            j += 1

        if fpdat_shift[idx] == 1:

            if fpdat[idx] == 0 and lookback == 0 and lookahead != 0:
                start_times.append(time_df[idx])

            elif lookahead - 1 == 0 and lookback != 0:

                if start_times:
                    start_time = start_times.pop()
                    pairs.append((start_time, time_df[idx]))

                else:
                    pass

            else:
                pass

        else:
            pass

        j = 0
        lookback = 0
        lookahead = 0

    if print_data is True:
        for idx in range(len(fpdat_shift) - 16):
            while j < 16:
                lookback = lookback + fpdat[idx + j]
                lookahead = lookahead + fpdat_shift[idx + j]
                j += 1

            if fpdat_shift[idx] == 1:
                if fpdat[idx] == 0 and lookback == 0 and lookahead != 0:
                    print(fpdat_shift[idx], idx, lookback, lookahead, "START >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", time_df[idx])
                elif lookahead - 1 == 0 and lookback != 0:
                    if start_times:
                        print(fpdat_shift[idx], idx, lookback, lookahead, "END ==========================", time_df[idx])
                    else:
                        print(fpdat_shift[idx], idx, lookback, lookahead, ">>> END ==========================", time_df[idx])                 
                else:
                    print(fpdat_shift[idx], idx, lookback, lookahead - 1, "######")
            else:
                print(fpdat_shift[idx], idx, lookback, lookahead - 1, "*")

            j = 0
            lookback = 0
            lookahead = 0
    
    total_difference = sum(end - start for start, end in pairs)
    tavg = total_difference/len(pairs)
    return tavg


def AverageSteadyVelocity(slow_time, fast_time, alpha, beta, length, belt_diff, measured_velocity):

    measured_velocity_avg = measured_velocity.mean()

    sinsum = np.sin(alpha) + np.sin(beta)

    sTravel = length*sinsum
    fTravel = (length*sinsum)- (fast_time*belt_diff)

    timesum = slow_time+fast_time
    sbWalkSpeed = ((2*length*sinsum)-(fast_time*belt_diff))/timesum

    # print(
    #     "========================================\n Slow Belt Time: ", slow_time, "\n",
    #     "Fast Belt Time: ", fast_time, "\n",
    #     "Time Sum:", timesum, "\n",
    #     "Belt Difference: ", belt_diff, "\n",
    #     "Slow Belt Distance Travelled:", sTravel, "\n",
    #     "Fast Belt Distance Travelled:", fTravel, "\n",
    #     "Total Belt Distance Travelled:", sTravel+fTravel, "\n",
    #     "Measured Average Steady Velocity: ", measured_velocity_avg, "\n",
    #     "Average Steady Velocity: ", sbWalkSpeed, "\n========================================",
    #     )

    print("=========================================\n Belt Difference: ", belt_diff, "\n",
          "Measured Average Steady Velocity: ", measured_velocity_avg, "\n",
          "Average Steady Velocity: ", sbWalkSpeed, "\n=========================================",
          )

    return measured_velocity_avg, sbWalkSpeed
