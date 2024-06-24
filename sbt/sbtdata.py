import pandas as pd
import numpy as np
import math

def ConstructData(timedata, fprdat, fpbdat):

    datapd = {

    'Time':timedata,
    'fprdat':fprdat,
    'fpbdat':fpbdat,

    }

    df = pd.DataFrame(datapd)
    df['fprdat_shift'] = df['fprdat'].shift(-16)
    df['fpbdat_shift'] = df['fpbdat'].shift(-16)

    fprdat = df['fprdat']
    fpbdat = df['fpbdat']
    fprdat_shift = df['fprdat_shift']
    fpbdat_shift = df['fpbdat_shift']

    df.to_csv('sbt_data', encoding="utf-8")

    return [fprdat, fprdat_shift, fpbdat, fpbdat_shift]

def findTime(fpdat, fpdat_shift):

    df = pd.read_csv('sbt_data')
    time_df = df['Time']

    stimelist = []
    etimelist = []

    idx = 1
    j = 0
    lookback = 0
    lookahead = 0

    fpdat_shift = fpdat_shift.copy()
    fpdat = fpdat.copy()

    fpdat_shift[fpdat_shift == 9.81] = 0
    fpdat_shift[fpdat_shift != 0] = 1

    fpdat[fpdat==9.81] = 0
    fpdat[fpdat!=0] = 1

    for idx in range(len(fpdat_shift)-16):

        while j < 16:
            lookback = lookback + fpdat[idx+j]
            lookahead = lookahead + fpdat_shift[idx+j]
            j+=1

        if fpdat_shift[idx] == 1:

            if fpdat[idx] == 0 and fpdat_shift[idx+1] == 1 and lookback == 0 and lookahead!=0:
                print(fpdat_shift[idx], idx,  lookback, lookahead,"START >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", time_df[idx])
                stimelist.append(time_df[idx])
                

            elif lookahead-1==0 and lookback!=0:
                print(fpdat_shift[idx], idx,  lookback, lookahead,"END ==========================", time_df[idx])
                etimelist.append(time_df[idx])
            else:
                print(fpdat_shift[idx], idx, lookback, lookahead-1,"######")
                pass

        else:   
            print(fpdat_shift[idx], idx, lookback, lookahead-1,"*")
            pass
            

        j = 0
        lookback = 0
        lookahead = 0

    # Create pairs of start and end times

    paired_list = list(zip(stimelist, etimelist))
    total_difference = sum(end - start for start, end in paired_list)
    tavg = total_difference/len(paired_list)
    return tavg


def AverageSteadyVelocity(slow_time, fast_time, alpha, beta, length, belt_diff):

    sinsum = np.sin(alpha) + np.sin(beta)

    sTravel = length*sinsum
    fTravel = (length*sinsum)- (fast_time*belt_diff)

    timesum = slow_time+fast_time
    sbWalkSpeed = ((2*length*sinsum)-(fast_time*belt_diff))/timesum
    print(
        "\n Slow Belt Time: ", slow_time, "\n",
        "Fast Belt Time: ", fast_time, "\n",
        "Time Sum:", timesum, "\n",
        "Belt Difference: ", belt_diff, "\n",
        "Slow Belt Distance Travelled:", sTravel, "\n",
        "Fast Belt Distance Travelled:", fTravel, "\n",
        "Total Belt Distance Travelled:", sTravel+fTravel, "\n",
        "Average Steady Velocity: ", sbWalkSpeed, "\n",
        )

