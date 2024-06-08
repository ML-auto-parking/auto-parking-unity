using System;
using System.Collections.Generic;
using AutonomousParking.Common.Enumerations;
using UnityEngine;

namespace AutonomousParking.Agents.Data
{
    [Serializable]
    // ParkingAgentRewardData 클래스는 주차 에이전트의 보상 관련 데이터를 저장합니다.
    public class ParkingAgentRewardData
    {
        [field: SerializeField]
        public float MaxRewardForInactivityPerStep { get; private set; }

        [field: SerializeField] // 목표까지의 거리를 줄였을 때 단계별로 주어지는 최대 보상입니다.
        public float MaxRewardForDecreasingDistanceToTargetPerStep { get; private set; }
        [field: SerializeField] // 목표까지의 거리를 줄였을 때 단계별로 주어지는 최대 보상입니다.
        public float MaxRewardForDecreasingLongDistanceToTargetPerStep { get; private set; }
        
        [field: SerializeField] // 목표까지의 각도를 줄였을 때 단계별로 주어지는 최대 보상입니다.
        public float MaxRewardForDecreasingAngleToTargetPerStep { get; private set; }

        [field: Header("Parking Rewards")]
        [field: SerializeField] // 주차 시작 단계에서 주어지는 최소 보상입니다.
        public float MinRewardForParkingPerStep { get; private set; }

        [field: SerializeField] // 주차 시작 단계에서 주어지는 최대 보상입니다.
        public float MaxRewardForParkingPerStep { get; private set; }

        [field: SerializeField] // 완벽한 주차에 대한 최소 보상입니다.
        public float MinRewardForPerfectParking { get; private set; }

        [field: SerializeField] // 완벽한 주차에 대한 최대 보상입니다.
        public float MaxRewardForPerfectParking { get; private set; }

        [field: Header("Collision Rewards")]
        [field: SerializeField] // 벽과의 충돌시 주어지는 보상입니다.
        public float RewardForWallCollisionEnter { get; private set; }

        [field: SerializeField] // 다른 차량과의 충돌시 주어지는 보상입니다.
        public float RewardForCarCollisionEnter { get; private set; }

        // 충돌 보상을 관리하는 딕셔너리입니다. 각 태그에 따라 다른 보상을 저장합니다.
        public Dictionary<Tag, float> CollisionRewards { get; private set; } = new();

        // Initialize 메서드는 충돌 보상을 초기화합니다.
        // 딕셔너리에 태그별 보상 값을 설정합니다.
        public void Initialize()
        {
            CollisionRewards[Tag.Wall] = RewardForWallCollisionEnter;
            CollisionRewards[Tag.Car] = RewardForCarCollisionEnter;
        }
    }
}
