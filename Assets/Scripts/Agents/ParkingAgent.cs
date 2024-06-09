using AutonomousParking.Agents.Components;
using AutonomousParking.Agents.Data;
using AutonomousParking.Car;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;
using UnityEngine.UI;
using ParkingManager;
using System.Collections.Generic;
using AutonomousParking.ParkingLot.ObjectPlacers;
using UnityEngine.UI;

namespace AutonomousParking.Agents
{   
    public class ParkingAgent : Agent
    {
        [field: SerializeField] public ParkingAgentData AgentData { get; private set; }
        [field: SerializeField] public ParkingAgentTargetTrackingData TargetTrackingData { get; private set; }
        [field: SerializeField] public ParkingAgentRewardData RewardData { get; private set; }
        public CarData CarData { get; set; }
        public ParkingAgentTargetData TargetData { get; set; }
        public ParkingAgentCollisionData CollisionData { get; set; }

        public ParkingLotEnteringCarPlacer AgentPlacer { get; set; }
        public ParkingLotAgentTargetPlacer TargetPlacer { get; set; }
        public ParkingLotParkedCarsPlacer ParkedCarsPlacer { get; set; }

        public ParkingAgentActionsHandler ActionsHandler { get; set; }
        public ParkingAgentMetricsCalculator MetricsCalculator { get; set; }
        public ParkingAgentRewardCalculator RewardCalculator { get; set; }
        public ParkingAgentObservationsCollector ObservationsCollector { get; set; }
        public ParkingAgentCollisionsHandler CollisionsHandler { get; set; }
        public ParkingAgentStatsRecorder StatsRecorder { get; set; }
        private RayPerceptionSensorComponent3D rayPerceptionSensor;
        public Text rewardText;
        public Text stepText;
        public override void Initialize()
        {
            rayPerceptionSensor = GetComponentInChildren<RayPerceptionSensorComponent3D>();
            var initializer = GetComponentInParent<ParkingAgentInitializer>();
            initializer.InitializeExternal(this);
            initializer.InitializeData(this);
            initializer.InitializeComponents(this);
            if (rayPerceptionSensor == null)
            {
                Debug.LogError("RayPerceptionSensorComponent3D is not assigned or found on the GameObject.");
            }
            else
            {
                Debug.Log("RayPerceptionSensorComponent3D successfully assigned.");
            }
        }

        public override void OnEpisodeBegin()
        {
            AgentData.Reset();
            CarData.Reset();
            
            ParkedCarsPlacer.Remove();
            ParkedCarsPlacer.Place();
            AgentPlacer.Place(AgentData.Transform);
            TargetPlacer.Place(TargetData.Transform, AgentData.Transform);

            MetricsCalculator.CalculateInitialTargetTrackingMetrics();
        }

        public override void CollectObservations(VectorSensor sensor)
        {
            ObservationsCollector.CollectAgentTransformObservations(sensor);
            // ObservationsCollector.CollectAgentVelocityObservations(sensor);

            ObservationsCollector.CollectTargetTransformObservations(sensor);
            ObservationsCollector.CollectAgentActionObservations(sensor);
            ObservationsCollector.CollectParkingSuccessObservations(sensor);
            // CheckRayCast();
        }

        public override void OnActionReceived(ActionBuffers actions)
        {
            ActionsHandler.HandleInputActions(actions);
            MetricsCalculator.CalculateTargetTrackingMetrics();
            AddReward(RewardCalculator.CalculateReward());

            // Debug.Log($"Step: {StepCount}, Reward: {RewardCalculator.CalculateReward()}, Cumulative Reward: {GetCumulativeReward()}"); // 누적 보상 출력
            rewardText.text = "Reward: " + GetCumulativeReward().ToString("F2");
            stepText.text = "Step: " + StepCount.ToString("F2");
            bool isNeededToEndEpisode = CollisionData.IsAnyCollision || TargetTrackingData.IsPerfectlyParked;
            bool isLastStep = AgentData.HasReachedMaxStep || isNeededToEndEpisode;

            if (isLastStep)
                StatsRecorder.RecordStats();

            if (isNeededToEndEpisode)
                EndEpisode();

            CollisionData.Reset();
        }

        public override void Heuristic(in ActionBuffers actionsOut)
        {
            ActionsHandler.HandleHeuristicInputDiscreteActions(actionsOut.DiscreteActions);
        }

        private void CheckRayCast()
        {
            if (rayPerceptionSensor == null)
            {
                Debug.LogError("RayPerceptionSensorComponent3D is not assigned.");
                return;
            }

            var rayOutputs = RayPerceptionSensor.Perceive(rayPerceptionSensor.GetRayPerceptionInput()).RayOutputs;
            int lengthOfRayOutputs = rayOutputs.Length;
            //Debug.Log(lengthOfRayOutputs);
            // Alternating Ray Order: it gives an order of
            // (0, -delta, delta, -2delta, 2delta, ..., -ndelta, ndelta)
            // index 0 indicates the center of raycasts
            for (int i = 0; i < lengthOfRayOutputs; i++)
            {
                GameObject goHit = rayOutputs[i].HitGameObject;
                if (goHit != null)
                {
                    var rayDirection = rayOutputs[i].EndPositionWorld - rayOutputs[i].StartPositionWorld;
                    var scaledRayLength = rayDirection.magnitude;
                    float rayHitDistance = rayOutputs[i].HitFraction * scaledRayLength;

                    // Print info:
                    string dispStr = "";
                    dispStr = dispStr + "__RayPerceptionSensor - HitInfo__:\r\n";
                    dispStr = dispStr + "GameObject name: " + goHit.name + "\r\n";
                    dispStr = dispStr + "Hit distance of Ray: " + rayHitDistance + "\r\n";
                    dispStr = dispStr + "GameObject tag: " + goHit.tag + "\r\n";
                    Debug.Log(i+":"+dispStr);
                }
            }
        }
    }
}
