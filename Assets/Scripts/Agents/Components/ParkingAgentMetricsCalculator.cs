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

        private int PastStepCountForDistance;
        private int PastStepCountForAngle;

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
            data.MaxAdditionalDistanceToTarget = CalculateDistanceToTarget();
            data.MaxAngleToTarget = 65f;
            // this.PastStepCountForDistance = 0;
            // this.PastStepCountForAngle = 0;
        }

        public void CalculateTargetTrackingMetrics()
        {
            data.DistanceToTarget = CalculateDistanceToTarget();
            data.NormalizedDistanceToTarget = CalculateNormalizedDistanceToTarget();
            
            data.AngleToTarget = CalculateAngleToTarget();
            data.NormalizedAngleToTarget = CalculateNormalizedAngleToTarget();
            data.NormalizedDistanceToTargetForAngle = CalculateNormalizedDistanceToTargetForAngle();
            data.NormalizedAdditionalDistanceToTarget = CalculateAdditionalNormalizedDistanceToTarget();
            data.NormalizedFacingDirectionToTarget = CalculateNormalizedFacingDirectionToTarget();

            data.IsParked = CalculateWhetherAgentIsParked();
            data.IsPerfectlyParked = CalculateWhetherAgentIsPerfectlyParked();
            data.IsGettingRewardForDecreasingAngleToTarget = CalculateWhetherToGetRewardForDecreasingAngleToTarget();

            // if (data.DistanceToTarget < data.MaxDistanceToTarget)
            // {
            //     if (data.DistanceToTarget >= data.MaxDistanceToTargetToGetRewardForDecreasingAngle){
            //         data.MaxDistanceToTarget = data.DistanceToTarget;
            //     } else {
            //         data.MaxDistanceToTarget = data.MaxDistanceToTargetToGetRewardForDecreasingAngle;
            //     }
            // }

            // Debug.Log("Angle to target: " + data.MaxDistanceToTargetToGetRewardForDecreasingAngle);
            // if (data.DistanceToTarget < data.MaxDistanceToTargetToGetRewardForDecreasingAngle && (agentData.StepCount - this.PastStepCountForAngle) > 100 && (data.AngleToTarget < data.MaxAngleToTarget))
            // {
            //     this.PastStepCountForAngle = agentData.StepCount;
            //     if (data.DistanceToTarget >= targetData.ParkingRadius){
            //         data.MaxDistanceToTargetToGetRewardForDecreasingAngle = data.DistanceToTarget;
            //     } else {
            //         data.MaxDistanceToTargetToGetRewardForDecreasingAngle = targetData.ParkingRadius;
            //     }
            // }
        }

        private float CalculateDistanceToTarget()
        {
            Vector3 agentPosition = agentData.Transform.position;
            Vector3 targetPosition = targetData.Transform.position;
            
            // y를 제외한 x와 z 좌표로 거리 계산
            float distance = Mathf.Sqrt(Mathf.Pow(agentPosition.x - targetPosition.x, 2) + Mathf.Pow(agentPosition.z - targetPosition.z, 2));
            return distance;
        }

        private float CalculateNormalizedDistanceToTarget() =>
            data.DistanceToTarget.Normalize(data.MaxDistanceToTarget, data.MinDistanceToTarget);

        private float CalculateAdditionalNormalizedDistanceToTarget() =>
            data.DistanceToTarget.Normalize(data.MaxAdditionalDistanceToTarget, data.MinDistanceToTarget);

        private float CalculateNormalizedAngleToTarget() {
            return data.AngleToTarget.Normalize(data.MaxAngleToTarget, data.MinAngleToTarget);
        }

        private float CalculateNormalizedDistanceToTargetForAngle() =>
            data.DistanceToTarget.Normalize(data.MaxDistanceToTargetToGetRewardForDecreasingAngle, data.MinDistanceToTarget);

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

        private float CalculateAngleToTarget()
        {
            float angle = Quaternion.Angle(agentData.Transform.rotation, targetData.Transform.rotation);
            return AdjustAngleForMultipleOrientations(angle);
        }

        private float CalculateNormalizedFacingDirectionToTarget()
        {
            // 에이전트의 전방 방향 벡터
            Vector3 agentForward = agentData.Transform.forward;

            // 에이전트의 위치에서 목표 지점까지의 방향 벡터
            Vector3 targetDirection = (targetData.Transform.position - agentData.Transform.position).normalized;

            // 두 벡터 사이의 각도 계산
            float angleToTarget = Vector3.Angle(agentForward, targetDirection);

            // 각도를 0과 1 사이의 값으로 정규화 (0도 또는 180도 -> 1, 90도 -> 0)
            float normalizedFacingDirection = Mathf.Abs(Mathf.Cos(Mathf.Deg2Rad * angleToTarget));

            return normalizedFacingDirection;
        }


        private bool CalculateWhetherToGetRewardForDecreasingAngleToTarget() =>
            Mathf.Abs(data.DistanceToTarget) <= data.MaxDistanceToTargetToGetRewardForDecreasingAngle;
    }
}