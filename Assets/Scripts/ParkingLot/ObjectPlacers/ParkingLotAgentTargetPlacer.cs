using System.Collections.Generic;
using AutonomousParking.Common.Extensions;
using AutonomousParking.ParkingLot.Data;
using UnityEngine;

namespace AutonomousParking.ParkingLot.ObjectPlacers
{
    public class ParkingLotAgentTargetPlacer : MonoBehaviour
    {
        [SerializeField] private ParkingLotData parkingLotData;

        public void Place(Transform target,List<Component> Empty, Transform agent)
        {
            Empty.AddRange(parkingLotData.CurrentlyAvailableParkingSpots);
            target.position=default;
            //Transform closestParkingSpot = FindClosestParkingSpot(parkingLotData.CurrentlyAvailableParkingSpots);
            //target.position = closestParkingSpot.position;
            //target.rotation = closestParkingSpot.rotation;

            //Transform FindClosestParkingSpot(IEnumerable<Component> availableParkingSpots) =>
                //availableParkingSpots.MinBy(parkingSpot => Vector3.Distance(agent.position, parkingSpot.position));
        }
    }
}