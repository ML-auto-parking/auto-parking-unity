using AutonomousParking.Agents.Data;
using System.Collections.Generic;
using UnityEngine;

namespace AutonomousParking.ParkingLot.Data
{
    // ParkingLotData 클래스는 주차장 데이터를 관리하는 컴포넌트입니다.
    public class ParkingLotData : MonoBehaviour
    {
        // 주차 공간에 대한 데이터를 저장하는 ParkingSpotData 객체입니다.
        [field: SerializeField] public ParkingSpotData ParkingSpotData { get; private set; }

        // 사용 가능한 주차 공간들의 Component 컴포넌트 목록입니다.
        public List<Component> AvailableParkingSpots { get; private set; }
        [field: SerializeField] public List<Component> AvailableHorizontalParkingSpots { get; private set; }
        [field: SerializeField] public List<Component> AvailableVerticalParkingSpots { get; private set; }


        // 현재 사용 가능한 주차 공간들의 목록입니다. 
        public List<Component> CurrentlyAvailableParkingSpots { get; private set; }
        public List<Component> CurrentlyAvailableHorizontalParkingSpots { get; private set; }
        public List<Component> CurrentlyAvailableVerticalParkingSpots { get; private set; }

        // Awake 메소드는 객체가 처음 활성화될 때 호출됩니다. 이 메소드는 AvailableParkingSpots의 복사본을 CurrentlyAvailableParkingSpots로 설정합니다.
        private void Awake(){
            AvailableParkingSpots = new List<Component>();
            AvailableParkingSpots.AddRange(AvailableHorizontalParkingSpots);
            AvailableParkingSpots.AddRange(AvailableVerticalParkingSpots);

            CurrentlyAvailableHorizontalParkingSpots = new List<Component>(AvailableHorizontalParkingSpots);
            CurrentlyAvailableVerticalParkingSpots = new List<Component>(AvailableVerticalParkingSpots);

            CurrentlyAvailableParkingSpots = new List<Component>();
        }

        // Reset 메소드는 주차장의 상태를 초기화합니다. 모든 주차 공간을 다시 사용 가능하게 설정합니다.
        public void Reset()
        {
            CurrentlyAvailableHorizontalParkingSpots.Clear();  // 현재 목록을 비웁니다.
            CurrentlyAvailableVerticalParkingSpots.Clear();  // 현재 목록을 비웁니다.
            CurrentlyAvailableParkingSpots.Clear();

            CurrentlyAvailableHorizontalParkingSpots.AddRange(AvailableHorizontalParkingSpots);
            CurrentlyAvailableVerticalParkingSpots.AddRange(AvailableVerticalParkingSpots);
            for(int i = 0; i<CurrentlyAvailableHorizontalParkingSpots.Count; i++)
            {
                CurrentlyAvailableHorizontalParkingSpots[i].GetComponent<ParkingSpotArea>().enabled=false;
                CurrentlyAvailableHorizontalParkingSpots[i].GetComponent<Collider>().enabled=false;

            }
        }
    }
}
