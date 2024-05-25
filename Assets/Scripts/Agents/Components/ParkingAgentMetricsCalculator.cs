﻿using AutonomousParking.Agents.Data;
using AutonomousParking.Common.Extensions;
using UnityEngine;

namespace AutonomousParking.Agents.Components
{
    public class ParkingAgentMetricsCalculator
    {
        private readonly ParkingAgentData agentData;
        private readonly ParkingAgentTargetTrackingData data;
        private readonly ParkingAgentTargetData targetData;

        public ParkingAgentMetricsCalculator(ParkingAgentData agentData, ParkingAgentTargetData targetData,
            ParkingAgentTargetTrackingData data)
        {
            this.agentData = agentData;
            this.targetData = targetData;
            this.data = data;
        }

        public void CalculateInitialTargetTrackingMetrics()
        {
            data.MaxDistanceToTarget = CalculateDistanceToTarget();
            data.MaxAngleToTarget = CalculateAngleToTarget();
        }

        public void CalculateTargetTrackingMetrics()
        {
            data.DistanceToTarget = CalculateDistanceToTarget();
            data.NormalizedDistanceToTarget = CalculateNormalizedDistanceToTarget();
            data.AngleToTarget = CalculateAngleToTarget();
            data.NormalizedAngleToTarget = CalculateNormalizedAngleToTarget();

            data.IsParked = CalculateWhetherAgentIsParked();
            data.IsPerfectlyParked = CalculateWhetherAgentIsPerfectlyParked();
            data.IsGettingRewardForDecreasingAngleToTarget = CalculateWhetherToGetRewardForDecreasingAngleToTarget();
        }

        private float CalculateDistanceToTarget() =>
            Vector3.Distance(agentData.Transform.position, targetData.Transform.position);

        private float CalculateNormalizedDistanceToTarget() =>
            data.DistanceToTarget.NormalizeWithNegative(data.MaxDistanceToTarget, data.MinDistanceToTarget);

        private float CalculateNormalizedAngleToTarget() =>
            data.AngleToTarget.Normalize(data.MaxAngleToTarget, data.MinAngleToTarget);

        private float AdjustAngleForMultipleOrientations(float angle)
        {
            float adjustedAngle = angle % 360;
            if (adjustedAngle > 180)
            {
                adjustedAngle = 360 - adjustedAngle;
            }
            if (adjustedAngle > 90)
            {
                adjustedAngle = 180 - adjustedAngle;
            }
            return adjustedAngle;
        }

        private bool CalculateWhetherAgentIsParked()
        {
            float adjustedAngle = AdjustAngleForMultipleOrientations(Mathf.Abs(CalculateAngleToTarget()));
            return Mathf.Abs(data.DistanceToTarget) <= targetData.ParkingRadius &&
                adjustedAngle <= targetData.ParkingAngle;
        }

        private bool CalculateWhetherAgentIsPerfectlyParked()
        {
            float adjustedAngle = AdjustAngleForMultipleOrientations(Mathf.Abs(CalculateAngleToTarget()));
            return Mathf.Abs(data.DistanceToTarget) <= targetData.PerfectParkingRadius &&
                adjustedAngle <= targetData.PerfectParkingAngle;
        }

        public float CalculateAngleToTarget()
        {
            float angle = Quaternion.Angle(agentData.Transform.rotation, targetData.Transform.rotation);
            return AdjustAngleForMultipleOrientations(angle);
        }


        private bool CalculateWhetherToGetRewardForDecreasingAngleToTarget() =>
            Mathf.Abs(data.DistanceToTarget) <= data.MaxDistanceToTargetToGetRewardForDecreasingAngle;
    }
}