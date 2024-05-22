using AutonomousParking.Agents.Data;
using AutonomousParking.Common.Extensions;
using System;
using Unity.MLAgents;
using UnityEngine;

namespace AutonomousParking.Agents.Components
{
    public class ParkingAgentRewardCalculator // 주차 에이전트 보상 계산
    {
        private readonly ParkingAgentCollisionData agentCollisionData; // 주차 에이전트 충돌 데이터
        private readonly ParkingAgentData agentData; // 주차 에이전트 데이터
        private readonly ParkingAgentRewardData rewardData; // 주차 에이전트 보상 데이터
        private readonly ParkingAgentTargetTrackingData targetTrackingData; // 주차 에이전트 타겟 추적 데이터
        // 생성자
        public ParkingAgentRewardCalculator(ParkingAgentCollisionData agentCollisionData, ParkingAgentData agentData,
            ParkingAgentRewardData rewardData, ParkingAgentTargetTrackingData targetTrackingData)
        {
            this.agentCollisionData = agentCollisionData;
            this.agentData = agentData;
            this.rewardData = rewardData;
            this.targetTrackingData = targetTrackingData;
        }

        public float CalculateReward() // 보상 계산 함수
        {

            // 효율성 보상 계산
            float reward = -1; // 활발하지 않음(적게 움직이거나 불필요한 행동을 하는 경우)에 대한 보상 계산
            // 정확도 보상 계산
            if (agentData.isInTargetArea)
            {
                reward += CalculateRewardForDecreasingDistanceToTarget(); // 타겟까지의 거리 감소에 대한 보상 계산
                //Debug.Log(CalculateRewardForDecreasingDistanceToTarget());
                if (targetTrackingData.IsGettingRewardForDecreasingAngleToTarget) // 타겟까지의 각도 감소에 대한 보상 계산
                //Debug.Log(reward += CalculateRewardForDecreasingAngleToTarget());
                    reward += CalculateRewardForDecreasingAngleToTarget();
            }
            
            if (agentCollisionData.IsAnyCollision) // 충돌에 대한 보상 계산
                reward += rewardData.CollisionRewards[agentCollisionData.CollisionTag];

            if (targetTrackingData.IsParked) // 주차 완료에 대한 보상 계산
            {
                reward += CalculateRewardForParking(); // 주차에 대한 보상 계산
                if (targetTrackingData.IsPerfectlyParked) // 완벽한 주차에 대한 보상 계산
                    reward += CalculateRewardForPerfectParking();
            }

            return reward;
        }

        // 타겟까지의 거리 감소에 따른 보상을 계산합니다.
        private float CalculateRewardForDecreasingDistanceToTarget() =>
            targetTrackingData.NormalizedDistanceToTarget * rewardData.MaxRewardForDecreasingDistanceToTargetPerStep;

        // 타겟까지의 각도 감소에 따른 보상을 계산합니다.
        private float CalculateRewardForDecreasingAngleToTarget() =>
            targetTrackingData.NormalizedAngleToTarget * rewardData.MaxRewardForDecreasingAngleToTargetPerStep;

        // 주차에 성공했을 때의 보상을 계산합니다. 보상은 주차 시작 가능 단계 범위를 기준으로 계산됩니다.
        private float CalculateRewardForParking() => agentData.StepCount
            .ChangeBounds(agentData.MaxStepToStartParking, agentData.MinStepToStartParking,
                rewardData.MinRewardForParkingPerStep, rewardData.MaxRewardForParkingPerStep);

        // 완벽하게 주차했을 때의 추가 보상을 계산합니다.
        private float CalculateRewardForPerfectParking() => agentData.StepCount
            .ChangeBounds(agentData.MaxStepToStartParking, agentData.MinStepToStartParking,
                rewardData.MinRewardForPerfectParking, rewardData.MaxRewardForPerfectParking);
    }
}