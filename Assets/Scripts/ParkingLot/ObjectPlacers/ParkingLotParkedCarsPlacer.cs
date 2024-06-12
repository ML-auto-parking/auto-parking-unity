using System.Collections.Generic;
using AutonomousParking.Car.Creation;
using AutonomousParking.Common.Extensions;
using AutonomousParking.ParkingLot.Data;
using UnityEngine;

namespace AutonomousParking.ParkingLot.ObjectPlacers
{
    // ParkingLotParkedCarsPlacer 클래스는 주차장에 자동차를 배치하는 기능을 담당합니다.
    public class ParkingLotParkedCarsPlacer : MonoBehaviour
    {
        // CarSpawner 객체에 대한 참조를 직렬화하여 인스펙터에서 설정할 수 있도록 합니다.
        [SerializeField] private CarSpawner carSpawner;
        // ParkingLotData 객체에 대한 참조를 직렬화하여 인스펙터에서 설정할 수 있도록 합니다.
        [SerializeField] private ParkingLotData parkingLotData;

        // Place 메소드는 주차장에 차량을 배치하는 로직을 수행합니다.
        public void Place()
        {
            // 랜덤으로 차를 배치할 주차 공간을 선택합니다.
            List<Transform> parkingSpotsToOccupy = PickRandomParkingSpotsToOccupy();
            // 각 주차 공간에 차량을 배치합니다.
            parkingSpotsToOccupy.ForEach(parkingSpot => carSpawner.Spawn(parkingSpot, parkingLotData.ParkingSpotData));

            // 랜덤으로 비워둔 곳을 다시 availableparkingspot 지점에 저장
            List<Transform> availableParkingSpots = parkingLotData.CurrentlyAvailableParkingSpots;
            availableParkingSpots.AddRange(parkingLotData.CurrentlyAvailableVerticalParkingSpots);
            availableParkingSpots.AddRange(parkingLotData.CurrentlyAvailableHorizontalParkingSpots);

            // PickRandomParkingSpotsToOccupy 메소드는 무작위로 차량을 배치할 주차 공간을 결정합니다.
            List<Transform> PickRandomParkingSpotsToOccupy()
            {
                // 현재 사용 가능한 주차 공간 목록을 가져옵니다.
                List<Transform> availableVerticalParkingSpots = parkingLotData.CurrentlyAvailableVerticalParkingSpots;
                List<Transform> availableHorizontalParkingSpots = parkingLotData.CurrentlyAvailableHorizontalParkingSpots;
                
                // 차량이 배치될 주차 공간 수를 계산합니다.
                int occupiedVerticalParkingSpotsCount = availableVerticalParkingSpots.Count - 1;
                int occupiedHorizontalParkingSpotsCount = availableHorizontalParkingSpots.Count;
                
                List<Transform> totalParkingSpotsToOccupy = availableVerticalParkingSpots.ExtractRandomItems(occupiedVerticalParkingSpotsCount);
                totalParkingSpotsToOccupy.AddRange(availableHorizontalParkingSpots.ExtractRandomItems(occupiedHorizontalParkingSpotsCount));

                // 랜덤으로 주차 공간을 추출합니다.
                return totalParkingSpotsToOccupy;
            }
        }

        // Remove 메소드는 모든 차량을 주차장에서 제거하고 주차장 데이터를 리셋합니다.
        public void Remove()
        {
            // CarSpawner를 통해 모든 차량을 제거합니다.
            // carSpawner.DeSpawnAll();
            // 주차장 데이터를 초기 상태로 복원합니다.
            // parkingLotData.Reset();
        }
    }
}
