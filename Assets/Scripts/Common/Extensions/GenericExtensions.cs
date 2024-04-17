using System;

namespace AutonomousParking.Common.Extensions
{
    // GenericExtensions는 다양한 유형의 데이터에 대한 확장 메서드를 제공하는 정적 클래스입니다.
    public static class GenericExtensions
    {
        // IsLessThan<T> 확장 메서드는 T 타입의 두 값(value, other)을 비교하여,
        // 'value'가 'other'보다 작은 경우 true를 반환합니다.
        // 이 메서드는 T가 IComparable<T> 인터페이스를 구현해야 사용할 수 있습니다.
        // IComparable<T> 인터페이스를 구현하는 타입은 CompareTo 메서드를 통해 정렬 또는 비교가 가능합니다.
        public static bool IsLessThan<T>(this T value, T other) where T : IComparable<T> =>
            // value.CompareTo(other)는 value가 other보다 작으면 음수, 같으면 0, 크면 양수를 반환합니다.
            // 여기서 default(int)는 0을 의미하므로, 반환된 값이 0보다 작은지를 검사합니다.
            value.CompareTo(other) < default(int);
    }
}
