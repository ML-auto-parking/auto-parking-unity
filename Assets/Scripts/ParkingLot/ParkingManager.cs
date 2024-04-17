using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace ParkingManager
{
    public class ParkingManager : MonoBehaviour
    {
        public List<Transform> ParkingSpot; // 주차 공간 위치
        public GameObject[] carPrefabs; // 차량 프리팹 할당
        public int parallelParkingCount; // 평행 주차 공간 개수
        private GameObject[] CarObjects;
        public List<Transform> TargetPlace;


        void Start()
        {
            ResetParkingLot();
        }
        void Initialize()
        {
            for (int i = 0; i < CarObjects.Length; i++)
            {
                Destroy(CarObjects[i]);
            }
            CarObjects = null;
        }

        void ResetParkingLot()
        {
            int firstEmptySpace = Random.Range(0, parallelParkingCount);
            int secondEmptySpace = Random.Range(parallelParkingCount, ParkingSpot.Count);

            while (firstEmptySpace == secondEmptySpace)
            {
                secondEmptySpace = Random.Range(0, ParkingSpot.Count);
            }

            CarObjects = new GameObject[ParkingSpot.Count];
            TargetPlace.Add(ParkingSpot[firstEmptySpace]);
            TargetPlace.Add(ParkingSpot[secondEmptySpace]);

            for (int i = 0; i < ParkingSpot.Count; i++)
            {
                if (i == firstEmptySpace || i == secondEmptySpace)
                {
                    continue;
                }

                Quaternion rotation = Quaternion.identity;

                // 평행 주차 공간에 차량 배치 시 90도 또는 270도 회전
                if (i < parallelParkingCount)
                {
                    float angle = Random.Range(0, 2) * 180 + 90; // 90도 또는 270도
                    rotation = Quaternion.Euler(0, angle, 0);
                }

                CarObjects[i] = Instantiate(carPrefabs[Random.Range(0, carPrefabs.Length)], ParkingSpot[i].position, rotation);
            }
        }
    }
}

