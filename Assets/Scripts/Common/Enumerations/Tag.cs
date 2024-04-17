using System;
using System.Collections.Generic;
using System.Linq;
using AutonomousParking.Common.Extensions;

namespace AutonomousParking.Common.Enumerations
{
    // 주차 시스템에서 객체를 식별하기 위한 태그 정의
    public class Tag
    {
        // 정적 필드로 Wall, Car 태그 정의
        public static readonly Tag Wall = new(nameof(Wall));
        public static readonly Tag Car = new(nameof(Car));

        // 태그를 문자열과 Tag 객체로 매핑하는 딕셔너리
        private static readonly Dictionary<string, Tag> tags =
            EnumExtensions.GetEnumeration<Tag>().ToDictionary(tag => tag.name);

        private readonly string name; // 태그 이름
        private Tag(string name) => this.name = name; // 생성자

        public static IReadOnlyList<Tag> List { get; } = tags.Values.ToList().AsReadOnly(); // 모든 태그의 리스트를 제공

        public override string ToString() => name; // 태그 이름을 문자열로 반환

        // 문자열을 Tag 객체로 변환하는 암시적 변환 연산자
        // 변환 시 태그가 존재하지 않으면 InvalidOperationException 예외를 발생시킴
        public static implicit operator Tag(string tag) 
        {
            if (!tags.ContainsKey(tag))
                throw new InvalidOperationException($"Tag {tag} is not exist. Define it in {nameof(Tag)} class.");
            return tags[tag];
        }
    }
}