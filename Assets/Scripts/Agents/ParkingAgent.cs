using AutonomousParking.Agents.Components;
using AutonomousParking.Agents.Data;
using AutonomousParking.Car;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;
using ParkingManager;
using System.Collections.Generic;
using AutonomousParking.ParkingLot.ObjectPlacers;
using AutonomousParking.ParkingLot.Data;

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
        //public ParkingAgentEmptyData EmptyData1 { get; set; }
        //public ParkingAgentEmptyData EmptyData2 { get; set; }

        public ParkingLotEnteringCarPlacer AgentPlacer { get; set; }
        public ParkingLotAgentTargetPlacer TargetPlacer { get; set; }
        public ParkingLotParkedCarsPlacer ParkedCarsPlacer { get; set; }
        //public ParkingLotAgentEmptyPlacer EmptyPlacer { get; set; }
        public ParkingAgentActionsHandler ActionsHandler { get; set; }
        public ParkingAgentMetricsCalculator MetricsCalculator { get; set; }
        public ParkingAgentRewardCalculator RewardCalculator { get; set; }
        public ParkingAgentObservationsCollector ObservationsCollector { get; set; }
        public ParkingAgentCollisionsHandler CollisionsHandler { get; set; }
        public ParkingAgentStatsRecorder StatsRecorder { get; set; }
        public List<Transform> EmptyCenter;
        public override void Initialize()
        {
            var initializer = GetComponentInParent<ParkingAgentInitializer>();
            initializer.InitializeExternal(this);
            initializer.InitializeData(this);
            initializer.InitializeComponents(this);
        }

        public override void OnEpisodeBegin()
        {
            AgentData.Reset();
            CarData.Reset();
            EmptyCenter.Clear();
            
            ParkedCarsPlacer.Remove();
            ParkedCarsPlacer.Place();
            AgentPlacer.Place(AgentData.Transform);
            //EmptyPlacer.Place(EmptyData1.Transform, EmptyData2.Transform);
            //Debug.Log(EmptyData1.Transform+"\n"+EmptyData2.Transform);
            TargetPlacer.Place(TargetData.Transform, EmptyCenter, AgentData.Transform);
            Debug.Log(TargetData.Transform.position+"\n"+EmptyCenter[0].position+"\n"+EmptyCenter[1].position);

            MetricsCalculator.CalculateInitialTargetTrackingMetrics();
        }

        public override void CollectObservations(VectorSensor sensor)
        {
            ObservationsCollector.CollectAgentTransformObservations(sensor);
            ObservationsCollector.CollectAgentVelocityObservations(sensor);

            ObservationsCollector.CollectTargetTransformObservations(sensor);
        }

        public override void OnActionReceived(ActionBuffers actions)
        {
            ActionsHandler.HandleInputActions(actions);
            MetricsCalculator.CalculateTargetTrackingMetrics();
            AddReward(RewardCalculator.CalculateReward());
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
    }
}
